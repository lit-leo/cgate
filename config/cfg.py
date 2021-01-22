"""Experiment setup configuration file.

Contains variables, that used for creating instances of the model, specifying the number of epochs,
optimizers applied, etc.
"""


import os
from pathlib import Path
from functools import partial
import torch
from torchvision.transforms import (ColorJitter, Compose, Normalize,
                                    RandomAffine, RandomRotation,
                                    RandomHorizontalFlip, RandomVerticalFlip,
                                    ToTensor, ToPILImage,
                                    Resize, RandomCrop)


############################# Experiment setup #############################

# Model params
N_TASKS = 5
IN_CH = 3
OUT_DIM = 2
USE_GUMBEL_SIGMOID = True # in case of False, pytorch's implementation of
                          # Gumbel-softmax is used with [logit, -logit] pairing
                          # to obtain binary decision for kernel to be used

ARCH = 'resnet18'
"""str: variable, specifying, which architecture use in the backbone

Currently available architectures:
    resnet18,
    SimpleCNN (as mentioned in the paper).
"""


CONV_CH = 512  # Represent the number of output channels for each layer in SimpleCNN, should be set to 512 for resnet-18
r"""int: the number of output channels

The paper mention the value of 100 for SimpleCNN. It should be set to 512 in case of resnet-18"""
if ARCH == 'resnet':
    assert CONV_CH == 512

EPOCHS_PER_TASK = 200
SPARSITY_PATIENCE_EPOCHS = 10
LAMBDA_SPARSE = 0.1

# Freezing specification
FREEZE_FIXED_PROC = False
r"""bool: freezing strategy

True == "freeze if kernel is in top k% most used among all";
False == " freeze if kernel was chosen with probability > thr

To replicate paper's results use False"""
FREEZE_TOP_PROC = 0.4  # used with "in top k%" strategy
FREEZE_PROB_THR = 0.001  # used with "was chosen with probability > thr" strategy

# Normalization specification
NORMALIZATION_IN_BACKBONE = 'BatchNorm2d'
assert NORMALIZATION_IN_BACKBONE in {'InstanceNorm2d', 'BatchNorm2d', None}
USE_BATCHNORM_GATES = True

# Class-incremental details
REHEARSE_ON_CORESET = False  # Not properly implemented, should be set as False
USE_TASK_CLF_LOSS = False  # Used in class-incremental setup, not properly implemented

"""The two following parameters, specify, whether the task being solved is known on validation/test
    or task classifier should be used to identify, which task is being solved"""
TASK_SUPERVISED_VALIDATION = True
TASK_SUPERVISED_TEST = True

CKPT_FREQ = 999  # Not implemented in pytorch backend, deprecated

# Data specifications
DATASET_NAME = 'CIFAR10'
TRUNCATE_SIZE = None  # Used to reduce the size of the dataset
UPSCALE_SIZE = 64
TRAIN_TRANSFORM = Compose([
    ToPILImage(),
    Resize(UPSCALE_SIZE),
    RandomCrop(UPSCALE_SIZE, padding=5),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
TEST_TRANSFORM = Compose([
    ToPILImage(),
    Resize(UPSCALE_SIZE),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# TRAIN_TRANSFORM = None
# TEST_TRANSFORM = None

"""The TASK_PAIRS param was made for Split MNIST dataset, in which it controlled, which classes will be paired together.
    For CIFAR-10 it still can be used to control the composition of tasks, but in this case one should consult
    torchvision_dataset.targets field, since numbers represent indexes in that array"""
TASK_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

# Training params
BATCH_SIZE = 64
GRADIENT_CLIP_VAL = 1

# partial because net.parameters() is missing
# OPT = partial(torch.optim.SGD,
#               lr=0.001,
#               momentum=0.9,
#               weight_decay=5e-4)

OPT = partial(torch.optim.Adam,
              lr=1e-3)
SCHEDULER = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=0.1, patience=20, verbose=True)

# Experiment metadata
EXPERIMENT_TAG = 'sigmoid_lambda1e-1_200ep'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
GPUS = 1 if torch.cuda.is_available() else 0  # Used for pytorch-lightning backend, currently deprecated

# Reproducibility
torch.manual_seed(0)
# torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

############################# Environment & stuff #############################

ROOT_DIR = Path(os.environ['ROOT_DIR'])

# Input data

DATA_ROOT = ROOT_DIR / 'data'
TRAIN_ROOT = DATA_ROOT / 'train'
VAL_ROOT = DATA_ROOT / 'val'
TEST_ROOT = DATA_ROOT / 'test'

# Results

RESULTS_ROOT = ROOT_DIR / 'results' / EXPERIMENT_TAG
SAMPLES_ROOT = RESULTS_ROOT / 'samples'
SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_ROOT = RESULTS_ROOT / 'checkpoints'
CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_NAME = CHECKPOINTS_ROOT / 'model.ckpt'
LOGGING_ROOT = RESULTS_ROOT / 'log'
