r"""Train routines.

This module contains various functions, which are aimed to structure the
training procedure of the Conditional Channel Gated Networks.
"""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import cfg
from src.data import prepare_dataset, get_loaders, set_task, get_coreset_loader


def train_one_task(model, train_loader, val_loader, logger=None):
    optimizer = cfg.OPT(model.parameters())
    scheduler = cfg.SCHEDULER(optimizer) if cfg.SCHEDULER else None
    for epoch_num in tqdm(range(cfg.EPOCHS_PER_TASK)):
        train_step(model, optimizer, scheduler, epoch_num, train_loader, logger)

        val_dict = val_step(model, epoch_num, val_loader, logger)

        if scheduler:
            scheduler.step(val_dict['val_loss'])

        if epoch_num % cfg.CKPT_FREQ == 0:
            torch.save(model.state_dict(), f'{cfg.CHECKPOINTS_ROOT}/task_{0}_ep{epoch_num}.pt')


def val_step(model, epoch_num, val_loader, logger=None):
    r"""
    Perform one validation epoch.
    Args:
        model: ChannelGatedCL
        epoch_num: int
        val_loader: torch.utils.data.DataLoader
        logger: tensorboardX logger

    Returns:
        tensorboard_logs dict
    """
    epoch_logs = []

    model.eval()
    for batch in val_loader:
        x, y, head_idx = batch
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)
        head_idx = head_idx.to(cfg.DEVICE)

        out, task_logits = model(x, head_idx, task_supervised_eval=cfg.TASK_SUPERVISED_VALIDATION)
        head_loss = F.cross_entropy(out, y)
        if cfg.TASK_SUPERVISED_VALIDATION:
            task_loss = 0
        else:
            task_loss = F.cross_entropy(task_logits, head_idx)
        val_loss = head_loss + task_loss

        bs = y.shape[0]
        val_acc = (F.softmax(out, dim=-1).argmax(dim=-1) == y).sum() / float(bs)

        epoch_logs.append({'loss': val_loss.detach().cpu(),
                           'acc': val_acc.detach().cpu()})

    avg_loss = torch.stack([x['loss'] for x in epoch_logs]).mean()
    avg_acc = torch.stack([x['acc'] for x in epoch_logs]).mean()

    tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}

    if logger:
        for k,v in tensorboard_logs.items():
            logger.add_scalar(k, v, epoch_num)

    return tensorboard_logs


def train_step(model, optimizer, scheduler, epoch_num, train_loader, logger=None):
    r"""
    Perform one training epoch.
    Args:
        model: ChannelGatedCL
        optimizer: optimizer from torch.optim
        scheduler: scheduler from torch.optim
        epoch_num: int
        train_loader: torch.utils.data.DataLoader
        logger: tensorboardX logger

    Returns:
        None
    """
    epoch_logs = []
    model.train()
    for batch in train_loader:
        x, y, head_idx = batch
        x = x.to(cfg.DEVICE)
        y = y.to(cfg.DEVICE)
        head_idx = head_idx.to(cfg.DEVICE)

        optimizer.zero_grad()
        out, task_logits = model(x, head_idx)

        head_loss = F.cross_entropy(out, y)
        if cfg.USE_TASK_CLF_LOSS:
            task_loss = F.cross_entropy(task_logits, head_idx)
        else:
            task_loss = torch.FloatTensor([0]).to(cfg.DEVICE)

        if epoch_num <= cfg.SPARSITY_PATIENCE_EPOCHS:
            sparsity_loss = torch.FloatTensor([0]).to(cfg.DEVICE)
        else:
            sparsity_loss = model.calc_sparcity_loss(head_idx).to(cfg.DEVICE)
        loss = head_loss + task_loss + sparsity_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_VAL)
        optimizer.step()

        bs = y.shape[0]
        train_acc = (F.softmax(out, dim=-1).argmax(dim=-1) == y).sum() / float(bs)

        epoch_logs.append({'loss': loss.detach().cpu(),
                            'acc': train_acc.detach().cpu(),
                            'head_loss': head_loss.detach().cpu(),
                            'task_loss': task_loss.detach().cpu(),
                            'sparse_loss': sparsity_loss.detach().cpu()})

    avg_loss = torch.stack([x['loss'] for x in epoch_logs]).mean()
    avg_acc = torch.stack([x['acc'] for x in epoch_logs]).mean()
    avg_head_loss = torch.stack([x['head_loss'] for x in epoch_logs]).mean()
    avg_task_loss = torch.stack([x['task_loss'] for x in epoch_logs]).mean()
    avg_sparse_loss = torch.stack([x['sparse_loss'] for x in epoch_logs]).mean()

    tensorboard_logs = {
        'train_loss': avg_loss,
        'train_acc': avg_acc,
        'head_loss': avg_head_loss,
        'task_loss': avg_task_loss,
        'sparse_loss': avg_sparse_loss
    }

    if logger:
        for k,v in tensorboard_logs.items():
            logger.add_scalar(k, v, epoch_num)


# Todo: remove pytorch-lightning handling
def perform_task_incremental_test(lit_model, N_tasks):
    r"""
    When all tasks has been trained, use this function to check,
        how quality changed/stayed the same in the process of learning new
        tasks

    The last column represents final quality after all tasks being fitted and
        relevant kernels for each task being frozen.

    Args:
        lit_model: ChannelGatedCL
        N_tasks: int, total number of tasks

    Returns:
        torch.Tensor, with row i representing quality on the i-th task
            and column j specifying the snapshot moment:
            the quality is checked after tasks 0..j had been fitted
            and relevant kernels frozen.
    """
    scores = torch.zeros((N_tasks, N_tasks), dtype=torch.float)

    datasets = prepare_dataset(dataset_name=cfg.DATASET_NAME,
                               train_transform=cfg.TRAIN_TRANSFORM,
                               val_transform=cfg.TEST_TRANSFORM,
                               test_transform=cfg.TEST_TRANSFORM,
                               truncate_size=cfg.TRUNCATE_SIZE,
                               task_pairs=cfg.TASK_PAIRS)
    dataloaders = get_loaders(*datasets, batch_size=cfg.BATCH_SIZE)
    train_data, val_data, test_data = datasets
    train_loader, val_loader, test_loader = dataloaders

    # if lit_model is a pytorch-lightning wrapper
    if hasattr(lit_model, 'model'):
        trainer = get_trainer()

        for task_fitted_num in range(0, N_tasks):
            load_after_next_task(lit_model, task_fitted_num)

            for prev_task_num in range(0, task_fitted_num + 1):
                set_task(prev_task_num, train_data, val_data, test_data)
                test_results = trainer.test(lit_model, test_dataloaders=test_loader)
                scores[prev_task_num, task_fitted_num] = test_results['test_acc']

        return scores
    else:
        for task_fitted_num in range(0, N_tasks):
            load_after_next_task(lit_model, task_fitted_num)

            for prev_task_num in range(0, task_fitted_num + 1):
                set_task(prev_task_num, train_data, val_data, test_data)
                test_results = val_step(lit_model, 0, val_loader)
                scores[prev_task_num, task_fitted_num] = test_results['val_acc']

        return scores


def load_after_next_task(lit_model, next_task_num):
    r"""
    if the model currently supports k tasks, appends for an upcoming task
        and loads proper checkpoint
    Args:
        lit_model: ChannelGatedCL
        next_task_num: int

    Returns:
        None
    """
    if next_task_num == 0:
        lit_model.load_model_state_dict(f'after_task_{next_task_num}.ckpt')
    else:
        lit_model.add_task()
        lit_model.load_model_state_dict(f'after_task_{next_task_num}.ckpt')


def load_after_many_tasks(lit_model, prev_task_num, next_task_num):
    r"""
    Loads checkpoint, corresponding to next_task_num,
        appending the model with parameters along the way
    Args:
        lit_model: ChannelGatedCL
        prev_task_num: int, currently loaded task
        next_task_num:

    Returns:

    """
    n_additions = next_task_num - prev_task_num
    for _ in range(n_additions):
        lit_model.add_task()

    lit_model.load_model_state_dict(f'after_task_{next_task_num}.ckpt')
