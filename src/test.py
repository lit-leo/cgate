r"""Test routines.

This module contains inference infrastructure for Conditional Channel Gated Network.
For the purpose of investigating forgetting now it is only used for conducting
    task-incremental test accuracy control.
"""
from config import cfg
from src.lit_models import LitChannelGatedCL

import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import prepare_dataset, get_loaders, set_task
from src.utils import get_trainer
from src.train_utils import perform_task_incremental_test
from src.model import  ChannelGatedCL


if __name__ == '__main__':
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



