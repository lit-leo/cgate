"""Pytorch-lightning wrappers.

Module contain wrapper classes, required for successful integration of model defined in src.model
into pytorch-lightning framework.

Was tested only for pytorch-lightning==0.8.5 and found incompatible with later versions. Currently deprecated.

    Typical usage example:
    CGCL = LitChannelGatedCL(N_tasks=5, in_ch=1, out_dim=2, lambda_sparse=10)
    set_task(0, train_data, val_data, test_data)
    coreset_loader = get_coreset_loader(train_data)
    trainer = get_trainer(callbacks=[ChannelGatedCLCalbacks(coreset_loader)],
                      checkpoint_path=None, last_epoch=0, epoches_to_learn=10)
    trainer.fit(CGCL, train_dataloader=train_loader, val_dataloaders=val_loader)
"""


import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm.auto import tqdm

from config import cfg
from src.model import ChannelGatedCL
from src.data import TaskedDataset


# TODO: underline difference between task_logits and head_idx
# TODO: find a better way to introduce sparsity patience for n epochs
class LitChannelGatedCL(pl.LightningModule):
    def __init__(self, in_ch, out_dim,
                 conv_ch=100,
                 sparsity_patience_epochs=20,
                 lambda_sparse=0.5,
                 freeze_fixed_proc=True,
                 freeze_top_proc=0.05,
                 freeze_prob_thr=0.8):
        super().__init__()

        self.model = ChannelGatedCL(in_ch, out_dim,
                                    conv_ch,
                                    sparsity_patience_epochs,
                                    lambda_sparse,
                                    freeze_fixed_proc,
                                    freeze_top_proc,
                                    freeze_prob_thr)

    def forward(self, x, head_idxs, task_supervised_eval=False):
        return self.model(x, head_idxs, task_supervised_eval)

    def get_gates_sparsity_stat(self):
        return self.model.get_gates_sparsity_stat()

    def calc_sparcity_loss(self, head_idx):
        return self.model.calc_sparcity_loss(head_idx)

    def enable_gates_firing_tracking(self):
        self.model.enable_gates_firing_tracking()

    def reset_gates_firing_tracking(self):
        self.model.reset_gates_firing_tracking()

    def update_freezed_kernels_idx(self, task_idx):
        self.model.update_freezed_kernels_idx(task_idx)

    def get_gates_firing_stat(self):
        return self.model.get_gates_firing_stat()

    def freeze_relevant_kernels(self, task_identifier):
        self.model.freeze_relevant_kernels(task_identifier)

    def reinitialize_irrelevant_kernels(self):
        self.model.reinitialize_irrelevant_kernels()

    def add_task(self):
        self.model.add_task()

    def save_model_state_dict(self, fname=''):
        if fname:
            torch.save(self.model.state_dict(), cfg.CHECKPOINTS_ROOT / fname)
        else:
            torch.save(self.model.state_dict(), cfg.CHECKPOINT_NAME)

    def load_model_state_dict(self, fname=''):
        if fname:
            self.model.load_state_dict(torch.load(cfg.CHECKPOINTS_ROOT / fname))
        else:
            self.model.load_state_dict(torch.load(cfg.CHECKPOINT_NAME))

    # ----------------- Model ends here -----------------#
    def on_train_start(self):
        self.first_epoch_ = self.trainer.current_epoch

    def training_step(self, batch, batch_nb):
        x, y, head_idx = batch
        out, task_logits = self(x, head_idx)

        head_loss = F.cross_entropy(out, y)
        if cfg.USE_TASK_CLF_LOSS:
            task_loss = F.cross_entropy(task_logits, head_idx)
        else:
            task_loss = torch.FloatTensor([0]).to(cfg.DEVICE)

        self.current_epoch = self.current_epoch - self.first_epoch_
        if self.current_epoch <= self.model.sparsity_patience_epochs:
            sparcity_loss = torch.FloatTensor([0]).to(cfg.DEVICE)
        else:
            sparcity_loss = self.calc_sparcity_loss(head_idx).to(cfg.DEVICE)
        loss = head_loss + task_loss + sparcity_loss

        bs = y.shape[0]
        train_acc = (F.softmax(out, dim=-1).argmax(dim=-1) == y).sum() / float(bs)

        # tensorboard_logs = {'train_loss': loss,
        #                     'train_acc': train_acc,
        #                     'head_loss': head_loss,
        #                     'task_loss': task_loss,
        #                     'sparse_loss': sparcity_loss}

        if self.current_epoch % cfg.CKPT_FREQ == 0:
            self.save_model_state_dict(fname=str(self.current_epoch))
        return {'loss': loss,
                'acc': train_acc,
                'head_loss': head_loss,
                'task_loss': task_loss,
                'sparse_loss': sparcity_loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_head_loss = torch.stack([x['head_loss'] for x in outputs]).mean()
        avg_task_loss = torch.stack([x['task_loss'] for x in outputs]).mean()
        avg_sparse_loss = torch.stack([x['sparse_loss'] for x in outputs]).mean()

        tensorboard_logs = {
            'train_loss': avg_loss,
            'train_acc': avg_acc,
            'head_loss': avg_head_loss,
            'task_loss': avg_task_loss,
            'sparse_loss': avg_sparse_loss
        }

        self.logger.log_metrics(tensorboard_logs, self.current_epoch)

        # for k, v in tensorboard_logs.items():
        #     self.logger.experiment.add_scalar(k, v.item(), self.current_epoch)

        # log training accuracy at the end of an epoch
        results = {}
        return results

    def validation_step(self, batch, batch_nb):
        x, y, head_idx = batch
        task_supervised_eval = cfg.TASK_SUPERVISED_VALIDATION
        out, task_logits = self(x, head_idx,
                                task_supervised_eval=task_supervised_eval)
        head_loss = F.cross_entropy(out, y)
        if task_supervised_eval:
            task_loss = 0
        else:
            task_loss = F.cross_entropy(task_logits, head_idx)
        val_loss = head_loss + task_loss

        bs = y.shape[0]
        val_acc = (F.softmax(out, dim=-1).argmax(dim=-1) == y).sum() / float(bs)
        return {'loss': val_loss, 'acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        self.logger.log_metrics(tensorboard_logs, self.current_epoch)
        return {}

    def test_step(self, batch, batch_nb):
        x, y, head_idx = batch
        task_supervised_eval = cfg.TASK_SUPERVISED_TEST
        out, task_logits = self(x, head_idx,
                                task_supervised_eval=task_supervised_eval)

        head_loss = F.cross_entropy(out, y)
        if task_supervised_eval:
            task_loss = 0
        else:
            task_loss = F.cross_entropy(task_logits, head_idx)
        test_loss = head_loss + task_loss

        bs = y.shape[0]
        test_task_clf_acc = (F.softmax(task_logits, dim=-1).argmax(dim=-1) == head_idx).sum() / float(bs)
        test_acc = (F.softmax(out, dim=-1).argmax(dim=-1) == y).sum() / float(bs)
        return {'test_loss': test_loss, 'test_acc': test_acc,
                'test_task_clf_acc': test_task_clf_acc}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_task_clf_acc = torch.stack([x['test_task_clf_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss,
                            'test_acc': avg_acc,
                            'test_task_clf_acc': avg_task_clf_acc}
        # self.logger.log_metrics(tensorboard_logs, self.current_epoch)
        # for k, v in tensorboard_logs.items():
        #     self.logger.experiment.add_scalar(k, v.item(), self.current_epoch)
        return {'log': tensorboard_logs}

    def configure_optimizers(self):
        opt = cfg.OPT(self.parameters())
        return opt

    def train_dataloader(self):
        # REQUIRED
        mnist = datasets.MNIST(root='data', train=True,
                               download=True, transform=ToTensor())
        tasked = TaskedDataset(mnist.data / 255., mnist.targets)
        tasked.set_task(0)
        loader = DataLoader(tasked, batch_size=128, num_workers=4,
                            shuffle=True)
        return loader

#     def val_dataloader(self):
#         # OPTIONAL
#         return DataLoader(MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
#                           num_workers=4, batch_size=256)

    def test_dataloader(self):
        # OPTIONAL
        mnist = datasets.MNIST(root='data', train=False,
                               download=True, transform=ToTensor())
        tasked = TaskedDataset(mnist.data / 255., mnist.targets)
        tasked.set_task(0)
        loader = DataLoader(tasked, batch_size=256, num_workers=4,
                            shuffle=False)
        return loader


class RehearseOnCoreset(Callback):
    def __init__(self, coreset_loader):
        super().__init__()

        self.coreset_loader = coreset_loader

    def on_epoch_end(self, trainer, pl_module):
        if self.coreset_loader:
            optimizer = trainer.optimizers[0]
            for x, y, task_idx in self.coreset_loader:
                x = x.to(cfg.DEVICE)
                y = y.to(cfg.DEVICE)
                task_idx = task_idx.to(cfg.DEVICE)
                out, task_logits = pl_module(x, task_idx)
                loss = F.cross_entropy(task_logits, task_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
#             print('Task Classifier rehearsed on the coreset!')


class ProgressBar(Callback):
    """Global progress bar.
    TODO: add progress bar for training, validation and testing loop.
    """

    def __init__(self, global_progress: bool = True,
                 leave_global_progress: bool = True):
        super().__init__()

        self.global_progress = global_progress
        self.global_desc = "Epoch: {epoch}/{max_epoch}"
        self.leave_global_progress = leave_global_progress
        self.global_pb = None

    def on_fit_start(self, trainer):
        desc = self.global_desc.format(epoch=trainer.current_epoch + 1,
                                       max_epoch=trainer.max_epochs)

        self.global_pb = tqdm(
            desc=desc,
            total=trainer.max_epochs,
            initial=trainer.current_epoch,
            leave=self.leave_global_progress,
            disable=not self.global_progress,
        )

    def on_fit_end(self, trainer):
        self.global_pb.close()
        self.global_pb = None

    def on_epoch_end(self, trainer, pl_module):

        # Set description
        desc = self.global_desc.format(epoch=trainer.current_epoch + 1,
                                       max_epoch=trainer.max_epochs)
        self.global_pb.set_description(desc)

        # Set logs and metrics
        # logs = pl_module.log
        # for k, v in logs.items():
        #     if isinstance(v, torch.Tensor):
        #         logs[k] = v.squeeze().item()
        # self.global_pb.set_postfix(logs)

        # Update progress
        self.global_pb.update(1)
