"""Conditional Channel Gated Networks for Task-Aware Continual Learning.

The original paper can be found on https://arxiv.org/pdf/2004.00070.pdf.
"""

from collections import OrderedDict
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm

from config import cfg
from src.blocks import (FreezingMethod,
                        GatedConvDownBlock,
                        GatedConvSameBlock,
                        GatedConvResBlock,
                        TaskAgnosticWrapper)


class ChannelGatedCL(nn.Module):
    r"""Class, which implements the logic of the Conditional Channel Gated Network.

    The model consists of:
        Backbone, which serves as feature extractor;
        Task Classifier, which is used in task-unsupervised (class-incremental) setup
            and identifies, which classification head should be used for particular object;
        Multihead Classifier, which has a separate classification head for each task.

    Currently only task-incremental setup is fully functional.
    """
    def __init__(self, in_ch, out_dim,
                 conv_ch=100,
                 sparsity_patience_epochs=20,
                 lambda_sparse=0.5,
                 freeze_fixed_proc=True,
                 freeze_top_proc=0.05,
                 freeze_prob_thr=0.8):
        r"""Class constructor
        Args:
            in_ch: int, the number of channels in the input
            out_dim: int, dimensionality of the output
            conv_ch: int, number of channels in each conv block,
                left as placeholder, its role is overtaken by cfg.CONV_CHANNELS
            sparsity_patience_epochs: int, the number of epochs,
                during which sparse loss will not be applied
            lambda_sparse: float, constant in sparsity loss
            freeze_fixed_proc: bool, freezing strategy
                True == "freeze if kernel is in top k% most used among all";
                False == " freeze if kernel was chosen with probability > thr
            freeze_top_proc: float, specifying the part of the capacity to be available for freezing
            freeze_prob_thr: float, specifying the minimal usage frequency to consider kernel relevant
        """
        super().__init__()

        # general model properties; conv_ch does not really affect anything
        self.in_ch = in_ch
        self.out_dim = out_dim
        self.conv_ch = conv_ch

        # connected with sparsity
        self.lambda_sparse = lambda_sparse
        self.sparsity_patience_epochs = sparsity_patience_epochs

        # connected with freezing
        self.freezing_method = FreezingMethod(freeze_fixed_proc, freeze_top_proc, freeze_prob_thr)
        self.freezing_enabled = False

        # model parts
        self.backbone = Backbone(self.in_ch, self.conv_ch, self.freezing_method)
        self.task_clf = TaskClassifier(self.conv_ch)
        self.multihead_clf = MultiHeadClassifier(self.conv_ch, self.out_dim)

    def forward(self, x: torch.Tensor, head_idxs, task_supervised_eval=True):
        x = self.backbone(x)
        task_logits = self.task_clf(x)

        if self.training or task_supervised_eval:
            # select candidate representations according to head_idxs
            selected_candidates = x[(torch.arange(x.shape[0]), head_idxs)]
            out = self.multihead_clf(selected_candidates, head_idxs)
        else:
            task_idxs = torch.argmax(F.softmax(task_logits, dim=-1), dim=-1)
            selected_candidates = self.select_candidates(x, task_idxs)

            out = self.multihead_clf(selected_candidates, task_idxs)

        return out, task_logits

    def get_gates_sparsity_stat(self):
        r"""
        Collect layer-wise sparseness statistics
        Returns:
            torch.FloatTensor, representing per-layer sparsity statistics
        """
        return self.backbone.get_gates_sparsity_stat()

    def enable_gates_firing_tracking(self):
        r"""
        Enable tracking of the gates firing frequency in all layers of backbone.
        Returns:
            None
        """
        self.backbone.enable_gates_firing_tracking()

    def reset_gates_firing_tracking(self):
        r"""
        Disable gates' firing frequency tracking and reset all calculated values
        Returns:
            None
        """
        self.backbone.reset_gates_firing_tracking()

    def get_gates_firing_stat(self):
        r"""
        Collect layer-wise frequencies, with which kernels were used since last tracking enablement
        Returns:
            List of tuples (frequencies, number_of_aggregations) for each layer.
                Normalized frequency can be calculated as frequencies / number_of_aggregations
        """
        return self.backbone.get_gates_firing_stat()

    def update_frozen_kernels_idx(self, task_idx):
        r"""
        Initiate the update of relevant kernels mask, which later can be used to freeze them.
        Args:
            task_idx: int, current task id, specifying, which part of firing_tracking tensors
                will be used for relevance calculation

        Returns:
            None
        """
        self.backbone.update_frozen_kernels_idx(task_idx)

    def freeze_relevant_kernels(self, task_identifier):
        r"""
        Initiate freezing of the relevant kernels according to the layers' relevant kernels masks
        Args:
            task_identifier: int, current task id

        Returns:
            None
        """
        self.backbone.freeze_relevant_kernels(task_identifier)

    def reinitialize_irrelevant_kernels(self):
        r"""
        Initiate reinitialization of the irrelevant kernels in each layer of the backbone
        Returns:
            None
        """
        self.backbone.reinitialize_irrelevant_kernels()

    def calc_sparcity_loss(self, head_idx):
        r"""
        Calculate sparsity objective, which encourages gates
            to choose minimal set of relevant kernels only, thus preserving
            unused capacity of the model for further tasks

        Important! Current implementation assumes, that during training
            batch contains only objects from one task

        Args:
            head_idx: torch.LongTensor, for each object in the batch specifying the task it belongs to
        Returns:
            sparsity_loss: float
        """
        # Assuming, that during training head_idx should be tensor with the same number
        task_idx = head_idx.unique()[0].cpu().detach().clone()
        stat = self.get_gates_sparsity_stat()
        num_layers = stat.shape[0]
        out = 0

        for i, layer_stat in enumerate(stat):
            actual_path_stat = layer_stat[task_idx]
            out += actual_path_stat

        out = out * self.lambda_sparse / num_layers
        return out

    def add_task(self):
        r"""
        Append constituents with parameters for new task
        Returns:
            None
        """
        self.backbone.add_task()
        self.task_clf.add_task()
        self.multihead_clf.add_task()

    def save_model_state_dict(self, fname=''):
        if fname:
            torch.save(self.state_dict(), cfg.CHECKPOINTS_ROOT / fname)
        else:
            torch.save(self.state_dict(), cfg.CHECKPOINT_NAME)

    def load_model_state_dict(self, fname=''):
        if fname:
            self.load_state_dict(torch.load(cfg.CHECKPOINTS_ROOT / fname,
                                            map_location=cfg.DEVICE))
        else:
            self.load_state_dict(torch.load(cfg.CHECKPOINT_NAME,
                                            map_location=cfg.DEVICE))


class Backbone(nn.Module):
    """
        Conv network composed of several GatedConvBlocks.
        Accepts an image tensor of shape batch_size x in_channels x H x W
        Assumes batch size to be > 1

        Produces tensor of shape batch_size x N_tasks x N_channels x H x W
    """

    def __init__(self, in_ch, conv_ch, freezing_method):

        super().__init__()
        self.N_tasks = 1
        self.in_ch = in_ch
        self.n_channels = conv_ch

        self.freezing_method = freezing_method

        if cfg.ARCH == 'resnet18':
            self.layers = self.create_resnet18()
        elif cfg.ARCH == 'SimpleCNN':
            self.layers = self.create_SimpleCNN()
        else:
            raise NotImplementedError()

    def create_gatedconv_unit(self, kind, in_ch, out_ch,
                              freezing_method: FreezingMethod,
                              stride=1,
                              conv_params: dict = None):
        r"""
        Return appropriate GatedConvBlock.

            parameters:

            conv_params : dict, containing kernel size, stride, dilation, etc
                currently not implemented for resblock.
        Args:
            kind: str, specify, which type of block to use:
                GatedConvBlock with or w/o maxpool at the end
                or resblock.
            in_ch: int, number of input channels
            out_ch: int, number of output channels
            freezing_method: src.blocks.FreezingMethod
            stride: int, left here specially for resblock construction
            conv_params: dict, containing kernel size, stride, dilation, etc
                currently not implemented for resblock.

        Returns:

        """
        if kind == 'down':
            net = nn.Sequential(OrderedDict(
                [('gated_conv', GatedConvDownBlock(in_ch, out_ch,
                                                   freezing_method,
                                                   conv_params=conv_params))])
            )
        elif kind == 'same':
            net = nn.Sequential(OrderedDict(
                [('gated_conv', GatedConvSameBlock(in_ch, out_ch, 
                                                   freezing_method,
                                                   conv_params=conv_params))])
            )
        elif kind == 'resblock':
            net = nn.Sequential(OrderedDict(
                [('gated_conv', GatedConvResBlock(in_ch, out_ch,
                                                  freezing_method,
                                                  stride=stride))])
            ) 
        else:
            raise NotImplemented
        return net

    def create_SimpleCNN(self):
        r"""
        Create SimpleCNN model. Check the paper for detailed description of the architecture.
        Returns:
            torch.nn.Sequential
        """
        layers = []
        layers += [self.create_gatedconv_unit('down',
                                              self.in_ch, self.n_channels,
                                              self.freezing_method,
                                              conv_params={
                                                'kernel_size': 3,
                                                'padding': 1
                                              })]
        layers += [self.create_gatedconv_unit('down',
                                              self.n_channels, self.n_channels,
                                              self.freezing_method,
                                              conv_params={
                                                'kernel_size': 3,
                                                'padding': 1
                                              })]
        layers += [self.create_gatedconv_unit('same',
                                              self.n_channels, self.n_channels,
                                              self.freezing_method,
                                              conv_params={
                                                'kernel_size': 3,
                                                'padding': 1
                                              })]
        return nn.Sequential(*layers)
    
    def create_resnet18(self):
        r"""
        Create gated version of ResNet-18
        Returns:
            torch.nn.Sequential
        """
        layers = []

        # initial 7x7 conv
        layers += [nn.Sequential(
            GatedConvSameBlock(self.in_ch, 64, self.freezing_method,
                               conv_params={
                                   'kernel_size': 7,
                                   'padding': 3,
                                   'stride': 2
                               }),
            TaskAgnosticWrapper(
                nn.Sequential(nn.MaxPool2d(kernel_size=2))
            )
        )]

        # conv_1 #
        layers += [nn.Sequential(
            GatedConvResBlock(64, 64, self.freezing_method,
                              stride=1),
        )]
        layers += [nn.Sequential(
            GatedConvResBlock(64, 64, self.freezing_method,
                              stride=1),

        )]

        # conv_2 #
        layers += [nn.Sequential(
            GatedConvResBlock(64, 128, self.freezing_method,
                              stride=2),

        )]

        layers += [nn.Sequential(
            GatedConvResBlock(128, 128, self.freezing_method,
                              stride=1),

        )]
        # conv_3 #
        layers += [nn.Sequential(
            GatedConvResBlock(128, 256, self.freezing_method,
                              stride=2),

        )]

        layers += [nn.Sequential(
            GatedConvResBlock(256, 256, self.freezing_method,
                              stride=1),

        )]
        # conv_4 #
        layers += [nn.Sequential(
            GatedConvResBlock(256, 512, self.freezing_method,
                              stride=2),

        )]

        layers += [nn.Sequential(
            GatedConvResBlock(512, 512, self.freezing_method,
                              stride=1),

        )]

        return nn.Sequential(*layers)

    def forward(self, x):
        # Create input duplicate for each task
        x = x[:, None, :, :, :].expand(-1, self.N_tasks, -1, -1, -1).clone()
        out = self.layers(x)

        return out

    def enable_gates_firing_tracking(self):
        r"""
        Enable tracking of the gates firing frequency for all gated layers.
        Returns:
            None
        """
        for gated_layer in self.layers:
            gated_layer[0].enable_gates_firing_tracking()

    def reset_gates_firing_tracking(self):
        r"""
        Disable gates' firing frequency tracking and reset all calculated values
        Returns:
            None
        """
        for gated_layer in self.layers:
            gated_layer[0].reset_gates_firing_tracking()

    def get_frozen_kernels_masks(self) -> list:
        """
        Aggregate frozen kernels masks across all layers in backbone.

        Important!: this function just retrieves masks in their current state

        Returns:
            list with masks
        """
        output = []
        for i, gated_layer in enumerate(self.layers):
            if hasattr(gated_layer[0], 'frozen_kernels_mask'):
                output.append(gated_layer[0].frozen_kernels_mask.detach().cpu())
            elif hasattr(gated_layer[0], 'frozen_kernels_mask2'):
                output.append(gated_layer[0].frozen_kernels_mask2.detach().cpu())
            else:
                raise NotImplementedError
        return output

    def get_gates_sparsity_stat(self):
        r"""
        Collect sparsity statistics for each gated layer
        Returns:
            torch.FloatTensor
        """
        out = []
        for gated_layer in self.layers:
            gates_sparse_objective = list(gated_layer[0].taskwise_sparse_objective)
            out.append(torch.stack(gates_sparse_objective, dim=0))
        return torch.stack(out, dim=0)

    def get_gates_firing_stat(self):
        r"""
        Collect layer-wise frequencies, with which kernels were used since last tracking enablement
        Returns:
            List of tuples (frequencies, number_of_aggregations) for each layer.

        Note:
            Normalized frequency can be calculated as frequencies / number_of_aggregations
        """
        out = []
        for gated_layer in self.layers:
            if hasattr(gated_layer[0], 'channels_firing_freq'):
                out.append((gated_layer[0].channels_firing_freq,
                            gated_layer[0].n_aggregations))
            elif hasattr(gated_layer[0], 'mask2_firing_freq'):
                out.append((gated_layer[0].mask2_firing_freq,
                            gated_layer[0].n_aggregations))
            else:
                raise NotImplementedError
        return out

    def freeze_relevant_kernels(self, task_id):
        """
            Adjusts weights shadowing of gated_layers' FreezableConv2d
        """
        for i, gated_layer in enumerate(self.layers):
            gated_layer[0].freeze_relevant_kernels(task_id)

    def reinitialize_irrelevant_kernels(self):
        r"""
        Initiate reinitialization of the irrelevant kernels in each layer of the backbone
        Returns:
            None
        """
        for gated_layer in self.layers:
            gated_layer[0].reinitialize_irrelevant_kernels()

    def add_task(self):
        r"""
        Append all gated layers with parameters for new task
        Returns:
            None
        """
        for i, gated_layer in enumerate(self.layers):
            gated_layer[0].add_task_path()
        self.N_tasks += 1


class TaskClassifier(nn.Module):
    """
        Module, that predicts task for given data during training
         to be task-aware during inference.
        Note, that all tensors in batch are considered to be of the same task.
    """

    def __init__(self, in_ch):
        super().__init__()
        self.N_tasks = 1
        self.in_ch = in_ch

        self.clf = nn.Sequential(
            nn.Linear(self.N_tasks * self.in_ch, 64),
            nn.ReLU(),
            nn.Linear(64, self.N_tasks)
        )

    def forward(self, x):
        # task-agnostic 4D AdaptiveAvgPool and concat
        bs, N_tasks, N_ch, H, W = x.shape
        # TODO: try bs, N_tasks * N_ch, H, W
        x = nn.AdaptiveAvgPool2d((1, 1))(x.reshape(bs * N_tasks, N_ch, H, W))
        x = x.reshape(bs, N_tasks * N_ch)
        logits = self.clf(x)
        return logits

    # TODO: change absolute indexing in self.clf to the string indexing
    # (e.g. self.clf['in_linear'] instead of self.clf[0])
    def add_task(self):
        """
            Broadens self.clf to accept more
            candidates with increased N_tasks
        """
        self.N_tasks += 1

        old_in_linear = self.clf[0].weight.data
        self.clf[0] = nn.Linear(self.N_tasks * self.in_ch, 64).to(old_in_linear)
        self.clf[0].weight.data[:, :(self.N_tasks - 1) * self.in_ch] = old_in_linear.data

        old_out_linear = self.clf[-1].weight.data
        self.clf[-1] = nn.Linear(64, self.N_tasks).to(old_out_linear)
        self.clf[-1].weight.data[:(self.N_tasks - 1), :] = old_out_linear.data


class MultiHeadClassifier(nn.Module):
    r"""
    Classifier with task-specific head, which yields final predictions.
    """
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.N_tasks = 1
        self.in_ch = in_ch
        self.out_dim = out_dim

        self.heads = nn.ModuleList(
            [self.create_single_head() for _ in range(self.N_tasks)])

    def create_single_head(self):
        # original from the paper 
        head_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.in_ch, self.out_dim),
        )
        # heavy one
#         head_fc = nn.Sequential(
#             nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(self.in_ch, self.out_dim),
#         )
        return head_fc

    def forward(self, x, head_idx):
        if x.ndim == 5:
            x = x.squeeze(1)

        # if batch contain only one task - only one head required,
        # do not separate elements
        if len(head_idx.unique()) == 1:
            active_head = self.heads[head_idx[0]]
            logits = active_head(x)
        else:
            active_heads = [self.heads[idx] for idx in head_idx.data.long()]
            # data[None, ...] serves for re-introducing batch_size dimention
            logits = [head(data[None, ...]) for head, data in zip(active_heads, x)]
            logits = torch.stack(logits, dim=1)[0]
        return logits

    def add_task(self):
        self.heads.append(self.create_single_head().to(self.heads[0][-1].weight))


# TODO: rewrite frozen_kernels_mask aggregation if save_freqs=True
# TODO: remove pytorch-lightning presence
def freeze_relevant_kernels(litmodel, val_loader, 
                            task_identifier: int, verbose=True,
                            save_freqs=False, save_fname=''):
    """
        Aggregate gates firing frequencies to choose, which kernels to freeze
        and which to reinitialize. Should be called after
        particular task has been fitted.
    """
    # check, if litmodel is an instance of pytorch-lightning wrapper
    lightning_model = hasattr(litmodel, 'model')
    litmodel.to(cfg.DEVICE)
    # Switch on gates firing stat aggregation
    litmodel.enable_gates_firing_tracking()
    
    # Do forward on validation, aggregate firing stat
    litmodel.model.eval() if lightning_model else litmodel.eval()

    if verbose:
        iterator = tqdm(val_loader)
    else:
        iterator = val_loader
    for x, y, task_idx in iterator:
        x = x.to(cfg.DEVICE)
        task_idx = task_idx.to(cfg.DEVICE)
        _, _ = litmodel(x, task_idx)

    # Enable freezing
    if lightning_model:
        litmodel.model.freezing_enabled = True
    else:
        litmodel.freezing_enabled = True
    litmodel.freeze_relevant_kernels(task_identifier)  
    litmodel.reinitialize_irrelevant_kernels()

    if save_freqs:
        stat = litmodel.model.get_gates_firing_stat() if lightning_model else litmodel.get_gates_firing_stat()
        frozen_masks = litmodel.model.backbone.get_frozen_kernels_masks() if lightning_model else litmodel.backbone.get_frozen_kernels_masks()
        if save_fname:
            fname = save_fname
        else:
            fname = datetime.now().strftime("%d_%m_%y_%H_%M")
        res = {'gates_freq': stat,
               'frozen_kernels_mask': frozen_masks}

        with open(fname + '.pickle', 'wb') as f:
            pickle.dump(res, f)

    # if next line is commented - don't forget to execute it manually later
    litmodel.reset_gates_firing_tracking()
    litmodel.model.train() if lightning_model else litmodel.train()
