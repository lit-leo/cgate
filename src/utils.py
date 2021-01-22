r"""Various utility functions
"""

import os

import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Callback

from config import cfg
from src.model import freeze_relevant_kernels
from src.data import prepare_dataset, get_loaders, set_task, get_coreset_loader


def disable_batchnorm_running_stat(lit_model):
    for layer_num in range(3):
        for gate in lit_model.model.backbone.layers[layer_num][0].gates:
            gate[3].track_running_stats = False         

            


                       

