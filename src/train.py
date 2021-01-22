r"""Train routines.

This module contains training infrastructure for Conditional Channel Gated Networks.
"""


import os

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm

from config import cfg
from src import experiment
from src.lit_models import LitChannelGatedCL, RehearseOnCoreset, ProgressBar
from src.model import freeze_relevant_kernels, ChannelGatedCL
from src.data import prepare_dataset, get_loaders, set_task, get_coreset_loader
from src.train_utils import train_one_task, perform_task_incremental_test

from tensorboardX import SummaryWriter


def train_torch_model(model):
    r"""
    Perform training of the Conditional Channel Gated Network.
    The torch part in the function's name stands here to distinguish it
        from pytorch-lightning based one, which was removed with decision to
        abandon this framework.
    Args:
        model: ChannelGatedCL

    Returns:
        None
    """
    if not os.path.exists(cfg.RESULTS_ROOT):
        os.mkdir(cfg.RESULTS_ROOT)

    logger = SummaryWriter(logdir=f'{str(cfg.LOGGING_ROOT)}/task_0')

    datasets = prepare_dataset(dataset_name=cfg.DATASET_NAME,
                               train_transform=cfg.TRAIN_TRANSFORM,
                               val_transform=cfg.TEST_TRANSFORM,
                               test_transform=cfg.TEST_TRANSFORM,
                               truncate_size=cfg.TRUNCATE_SIZE,
                               task_pairs=cfg.TASK_PAIRS)
    dataloaders = get_loaders(*datasets, batch_size=cfg.BATCH_SIZE)
    train_data, val_data, test_data = datasets
    train_loader, val_loader, test_loader = dataloaders

    # 0-th task fit
    task_num = 0
    set_task(task_num, train_data, val_data, test_data)
    train_one_task(model, train_loader, val_loader, logger)

    save_fname = f'{cfg.RESULTS_ROOT}/{cfg.DATASET_NAME}_task_{task_num}'
    freeze_relevant_kernels(model, val_loader,
                            task_identifier=task_num, verbose=False,
                            save_freqs=True,
                            save_fname=save_fname)
    model.save_model_state_dict(f'after_task_{0}.ckpt')
    logger.close()

    for task_num in range(1, cfg.N_TASKS):
        logger = SummaryWriter(logdir=f'{str(cfg.LOGGING_ROOT)}/task_{task_num}')
        set_task(task_num, train_data, val_data, test_data)
        model.add_task()

        train_one_task(model, train_loader, val_loader, logger)

        save_fname = f'{cfg.RESULTS_ROOT}/{cfg.DATASET_NAME}_task_{task_num}'
        freeze_relevant_kernels(model, val_loader,
                                task_identifier=task_num,
                                save_freqs=True, verbose=False,
                                save_fname=save_fname)
        model.save_model_state_dict(f'after_task_{task_num}.ckpt')
        logger.close()
    torch.save(model.state_dict(), f'{cfg.CHECKPOINTS_ROOT}/after_task_{task_num}.ckpt')


# Todo: remove pytorch-lightning handling
def aggregate_firing_stat_on_data(litmodel, data_loader, verbose=False):
    r"""
    Use data from data_loader to retrieve fates firing statistics for
        visualization or later analysis.
    Args:
        litmodel: ChannelGatedCL
        data_loader: torch.utils.data.DataLoader
        verbose: bool, specifying usage of tqdm

    Returns:
        Gates firing statistics: list of tuples (frequencies, number_of_aggregations)
            for each layer.

    Note:
        Normalized frequency can be calculated as frequencies / number_of_aggregations
    """
    # check, if litmodel is an instance of pytorch-lightning wrapper
    lightning_model = hasattr(litmodel, 'model')
    litmodel.to(cfg.DEVICE)
    litmodel.enable_gates_firing_tracking()

    litmodel.model.eval() if lightning_model else litmodel.eval()
    if verbose:
        iterator = tqdm(data_loader)
    else:
        iterator = data_loader
    for x, y, task_idx in iterator:
        x = x.to(cfg.DEVICE)
        task_idx = task_idx.to(cfg.DEVICE)
        _, _ = litmodel(x, task_idx)

    stat = litmodel.model.get_gates_firing_stat().copy() if lightning_model else litmodel.get_gates_firing_stat().copy()

    litmodel.reset_gates_firing_tracking()
    litmodel.model.train() if lightning_model else litmodel.train()
    return stat


if __name__ == '__main__':
    experiment.init()

    model = ChannelGatedCL(in_ch=cfg.IN_CH, out_dim=cfg.OUT_DIM,
                           conv_ch=cfg.CONV_CH,
                           sparsity_patience_epochs=cfg.SPARSITY_PATIENCE_EPOCHS,
                           lambda_sparse=cfg.LAMBDA_SPARSE,
                           freeze_fixed_proc=cfg.FREEZE_FIXED_PROC,
                           freeze_top_proc=cfg.FREEZE_TOP_PROC,
                           freeze_prob_thr=cfg.FREEZE_PROB_THR).to(cfg.DEVICE)

    train_torch_model(model)

    model = ChannelGatedCL(in_ch=cfg.IN_CH, out_dim=cfg.OUT_DIM,
                           conv_ch=cfg.CONV_CH,
                           sparsity_patience_epochs=cfg.SPARSITY_PATIENCE_EPOCHS,
                           lambda_sparse=cfg.LAMBDA_SPARSE,
                           freeze_fixed_proc=cfg.FREEZE_FIXED_PROC,
                           freeze_top_proc=cfg.FREEZE_TOP_PROC,
                           freeze_prob_thr=cfg.FREEZE_PROB_THR).to(cfg.DEVICE)

    task_incremental_acc = perform_task_incremental_test(model, cfg.N_TASKS)
    torch.save(task_incremental_acc, cfg.RESULTS_ROOT / 'task_incremental_acc.pt')
    print('\n---- Task incremental accuracies ----')
    print(task_incremental_acc)
