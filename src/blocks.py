"""Gated convolutional layers and resblock.

This module incorporates channel gated versions of regular Conv-BN-Relu and Residual Blocks. This versions support freezing of
relevant kernels; reinitialization of irrelevant ones; aggregation of freezing statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from src.buffer_container import BufferList
from src.freezable_layers import FreezableInstanceNorm2d, FreezableBatchNorm2d, FreezableConv2d


class FreezingMethod:
    r"""
    Class, specifying which method is used to calculate
        the relevance of convolutional kernels and thus controlling the
        freezing process.

    More information can be found in config/cfg.py
    """
    def __init__(self, freeze_fixed_proc=True, freeze_top_proc=0.05, freeze_prob_thr=0.8):
        super().__init__()
        self.freeze_fixed_proc = freeze_fixed_proc  # True == "in top k%"; False == "firied with probability > thr"
        self.freeze_top_proc = freeze_top_proc  # used with "in top k%" strategy
        self.freeze_prob_thr = freeze_prob_thr  # used with "firied with probability > thr" strategy


def create_freezable_bn(out_ch):
    r"""Return normalization layer according to cfg
    This function exists for correct backward compatibility
            with absence of cfg.NORMALIZATION_IN_BACKBONE in older configs
    """
    try:
        if cfg.NORMALIZATION_IN_BACKBONE == 'InstanceNorm2d':
            norm_class = FreezableInstanceNorm2d(out_ch)
        elif cfg.NORMALIZATION_IN_BACKBONE == 'BatchNorm2d':
            norm_class = FreezableBatchNorm2d(out_ch)
        elif cfg.NORMALIZATION_IN_BACKBONE is None:
            norm_class = nn.Identity()
        else:
            raise NotImplementedError
    except AttributeError:
        if cfg.USE_BATCHNORM_BACKBONE:
            norm_class = FreezableBatchNorm2d(out_ch)
        else:
            norm_class = nn.Identity()
    return norm_class


# Todo: finish sigmoid sampling
def gumbel_sigmoid(logits, tau):
    r"""Straight-through gumbel-sigmoid estimator
    """
    gumbels = -torch.empty_like(logits,
                                memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.sigmoid()
    
    # Straight through.
    y_hard = (y_soft > 0.5).long()
    ret = y_hard - y_soft.detach() + y_soft
    
    return ret


class GatedConvResBlock(nn.Module):
    r"""Gated convolution residual block for N tasks.
    Assumes data shape of batch_size x N_tasks x N_channels x H x W
    Assumes batch size to be > 1
    """

    def __init__(self, in_ch, out_ch, freezing_method,
                 stride=1,
                 aggregate_firing=False,
                 N_tasks=1,
                 ):
        super().__init__()

        self.N_tasks = N_tasks
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.use_opt_path = in_ch != out_ch

        self.conv2d_main1 = FreezableConv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=self.stride)
        self.conv2d_main2 = FreezableConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.conv2d_opt = FreezableConv2d(in_ch, out_ch, kernel_size=1, stride=self.stride)

        # Freezable task-specific normalizations
        self.fbns_main1 = nn.ModuleList([create_freezable_bn(self.out_ch) for _ in range(self.N_tasks)])
        self.fbns_main2 = nn.ModuleList([create_freezable_bn(self.out_ch) for _ in range(self.N_tasks)])
        self.fbns_opt = nn.ModuleList([create_freezable_bn(self.out_ch) for _ in range(self.N_tasks)])

        self.gates = nn.ModuleList([self.create_gate_fc() for _ in range(self.N_tasks)])

        # Variable to store sparse loss
        self.taskwise_sparse_objective = BufferList([torch.empty((1))] * self.N_tasks)

        # aggregates frequencies with which kernels were chosen
        self.aggregate_firing = aggregate_firing
        self.mask1_firing_freq = torch.zeros((self.N_tasks, self.out_ch))
        self.mask2_firing_freq = torch.zeros((self.N_tasks, self.out_ch))

        # TODO: rework stats aggregation:
        # This one is left only for backward compatability with
        # backbone.get_gates_firing_stat()
        # gotta rework it since 2 stats here are available.
        self.n_aggregations = torch.zeros((self.N_tasks))  # for calculating probabilities correctly

        self.freezing_method = freezing_method
        
        self.register_buffer('frozen_kernels_mask1',
                             torch.zeros((self.out_ch), dtype=int))
        self.register_buffer('frozen_kernels_mask2',
                             torch.zeros((self.out_ch), dtype=int))

    def create_gate_fc(self):
        gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.in_ch, 16),
            nn.BatchNorm1d(16, track_running_stats=False) if cfg.USE_BATCHNORM_GATES else nn.Identity(),
            nn.ReLU(),
            nn.Linear(16, self.out_ch * 2)  # output will be split into 2 masks
        )
        return gate_fc

    def add_task_path(self):
        r"""Add task path to the block to handle new upcoming task"""
        r"""Following .to(some_weight) operation is to ensure, that all the gates and fbns are on the same device
           This method, however, will not work with fbns, if affine=False, so be aware of that."""
        self.gates.append(self.create_gate_fc().to(cfg.DEVICE))
        self.fbns_main1.append(create_freezable_bn(self.out_ch).to(cfg.DEVICE))
        self.fbns_main2.append(create_freezable_bn(self.out_ch).to(cfg.DEVICE))
        self.fbns_opt.append(create_freezable_bn(self.out_ch).to(cfg.DEVICE))

        self.taskwise_sparse_objective.append(torch.empty((1)))


        self.mask1_firing_freq = torch.cat([self.mask1_firing_freq, 
                                            torch.zeros((1, self.out_ch))], 0)
        self.mask2_firing_freq = torch.cat([self.mask2_firing_freq, 
                                            torch.zeros((1, self.out_ch))], 0)
        self.n_aggregations = torch.cat([self.n_aggregations,
                                        torch.zeros((1))], 0)
        self.N_tasks += 1

    def enable_gates_firing_tracking(self):
        self.aggregate_firing = True

    def reset_gates_firing_tracking(self):
        self.aggregate_firing = False
        self.mask1_firing_freq = torch.zeros((self.N_tasks, self.out_ch))
        self.mask2_firing_freq = torch.zeros((self.N_tasks, self.out_ch))
        self.n_aggregations = torch.zeros((self.N_tasks))

    def aggregate_channels_firing_stat(self, mask1, mask2, task_idx):
        """
            Sums up frequencies of choosing kernels among batches
            during validation or test.

            Attributes:
            channels_mask - binary mask
        """
        self.mask1_firing_freq[task_idx] += mask1.float().mean(dim=(0, -1, -2)).detach().cpu()
        self.mask2_firing_freq[task_idx] += mask2.float().mean(dim=(0, -1, -2)).detach().cpu()
        self.n_aggregations[task_idx] += 1 

    def update_relevant_kernels(self, task_id):
        """
            Updates relevant kernels according to each gate-path i.e. task-path
        """
        if self.freezing_method.freeze_fixed_proc:
            k = int(self.out_ch * self.freezing_method.freeze_top_proc)
            aggregated_times = self.n_aggregations[task_id]
            threshold = self.freezing_method.freeze_prob_thr * aggregated_times

            mask1_stat = self.mask1_firing_freq[task_id].clone()
            mask2_stat = self.mask2_firing_freq[task_id].clone()
            n_relevant_1 = (mask1_stat > threshold).long().sum()
            n_relevant_2 = (mask2_stat > threshold).long().sum()
            # Todo: Account only for previously unfrozen kernels, silly!
            if n_relevant_1 > k:
                print(f'Not enough capacity for relevant kernels in mask1: {n_relevant_1}/{k}')
                idx_to_freeze_mask1 = torch.topk(mask1_stat, k, dim=-1)[1]
            else:
                idx_to_freeze_mask1 = torch.topk(mask1_stat, n_relevant_1, dim=-1)[1]

            if n_relevant_2 > k:
                print(f'Not enough capacity for relevant kernels in mask2: {n_relevant_2}/{k}')
                idx_to_freeze_mask2 = torch.topk(mask2_stat, k, dim=-1)[1]
            else:
                idx_to_freeze_mask2 = torch.topk(mask2_stat, n_relevant_2, dim=-1)[1]
        else:
            mask1_stat = self.mask1_firing_freq[task_id]
            mask2_stat = self.mask2_firing_freq[task_id]
            aggregated_times = self.n_aggregations[task_id]
            idx_to_freeze_mask1 = mask1_stat > self.freezing_method.freeze_prob_thr * aggregated_times
            idx_to_freeze_mask2 = mask2_stat > self.freezing_method.freeze_prob_thr * aggregated_times
        
        # aggregated mask becomes non-binary, but this does not interfere
        # with the logic of self.freeze_relevant_kernels()
        # and underlines the relevances of the kernels once more
        self.frozen_kernels_mask1[idx_to_freeze_mask1] += 1
        self.frozen_kernels_mask2[idx_to_freeze_mask2] += 1

    def freeze_relevant_kernels(self, task_id):
        r"""
        Initiate freezing of the relevant kernels
        Args:
            task_id: int, current task, which usage statistics will be used to
                calculate relevancy

        Returns:
            None
        """
        self.update_relevant_kernels(task_id)
        self.conv2d_main1.freeze(self.frozen_kernels_mask1.clamp(0, 1))
        self.conv2d_main2.freeze(self.frozen_kernels_mask2.clamp(0, 1))
        self.conv2d_opt.freeze(self.frozen_kernels_mask2.clamp(0, 1))

        r"""During training of task t only self.fbns_*[t] tracked relevant statistics, 
            therefore they are the only elements to freeze"""
        if cfg.NORMALIZATION_IN_BACKBONE:
            self.fbns_main1[task_id].freeze(self.frozen_kernels_mask1.clamp(0, 1))
            self.fbns_main2[task_id].freeze(self.frozen_kernels_mask2.clamp(0, 1))
            self.fbns_opt[task_id].freeze(self.frozen_kernels_mask2.clamp(0, 1))

    def reinitialize_irrelevant_kernels(self):
        r"""
        Invoke all freezable classes to reinitialize unfrozen kernels
        """

        self.conv2d_main1.reinit_unfrozen()
        self.conv2d_main2.reinit_unfrozen()
        self.conv2d_opt.reinit_unfrozen()

        r"""Despite only self.fbns_*[t] were properly frozen, all fbns should reinit irrelevant parameters, 
            according to their is_frozen masks"""
        if cfg.NORMALIZATION_IN_BACKBONE:
            for fbn_main1, fbn_main2, fbn_opt in zip(self.fbns_main1, self.fbns_main2, self.fbns_opt):
                fbn_main1.reinit_unfrozen()
                fbn_main2.reinit_unfrozen()
                fbn_opt.reinit_unfrozen()

    def sample_channels_mask(self, logits):
        """
            Samples binary mask to select
            relevant output channel of the convolution

            Attributes:
            logits - logprobabilities of the bernoully variables
                for each output channel of the convolution to be selected
        """
        if self.training:
            if cfg.USE_GUMBEL_SIGMOID:
                channels_mask = gumbel_sigmoid(logits, tau=2/3)
            else:
                bernoully_logits = torch.stack([logits, -logits], dim=0)
                channels_mask = F.gumbel_softmax(bernoully_logits, tau=2/3, hard=True, dim=0)[0]
        else:
            channels_mask = (logits > 0).long()
        return channels_mask

    def compute_masks(self, x, gate_fc, task_idx):
        """
            Performs selection of the output channels for the given task.

            Attributes:
            x - input tensor
            gate_fc - sequential model, provides logprobabilities for each output channel of the convolution
            task_idx - int label of the task path; used for gate firing aggregation
        """
        logits = gate_fc(x) # shape of [batch_size, 2 * self.out_ch]
        mask = self.sample_channels_mask(logits) # for both conv_blocks simultaneously
        # expand last 2 dims for channel-level elementwise multiplication
        mask = mask[:, :, None, None]
        # separate masks for the first and second main_conv, dim0 = batch_size
        mask_1, mask_2 = mask[:, :self.out_ch], mask[:, self.out_ch:]

        self.taskwise_sparse_objective[task_idx] = mask.float().mean().reshape(1)

        # TODO: add separate aggregation of masks
        if self.aggregate_firing:
            self.aggregate_channels_firing_stat(mask_1, mask_2, task_idx)

        return mask_1, mask_2

    def forward(self, x):
        # permute batch_size and N_tasks dims for task-wise iterations
        taskwise_input = x.permute(1, 0, 2, 3, 4)
        first_conv_masks, second_conv_masks = [], []

        for task_idx, (task_input, task_gate) in enumerate(zip(taskwise_input, self.gates)):
            mask_1, mask_2 = self.compute_masks(task_input, task_gate, task_idx)
            first_conv_masks.append(mask_1)
            second_conv_masks.append(mask_2)

        # stack and permute to batch-wise order for correct multiplication
        first_conv_masks = torch.stack(first_conv_masks, dim=1)
        second_conv_masks = torch.stack(second_conv_masks, dim=1)

        # task-agnostic conv2D+relu over 5D tensor
        bs, N_tasks, N_ch, H, W = x.shape
        task_agnostic_input = x.reshape(bs * N_tasks, N_ch, H, W)

        out1 = self.conv2d_main1(task_agnostic_input)
        N_ch, H, W = out1.shape[-3:]
        r"""applying BN is a tricky procedure:
            every task should be considered separately in order not to calculate statistics over batch of several tasks
            thus creating gradient leak into the gates, which outputs are ignored at the moment"""
        after_fbn_main1 = []
        for task_input, task_fbn_main1 in zip(out1.reshape(bs, N_tasks, N_ch, H, W).transpose(0, 1), self.fbns_main1):
                after_fbn_main1.append(F.relu(task_fbn_main1(task_input)))
        after_fbn_main1 = torch.stack(after_fbn_main1, dim=1)

        # apply first mask
        after_fbn_main1 *= first_conv_masks

        # task-agnostic conv2D over 5D tensor
        task_agnostic_out1 = after_fbn_main1.reshape(bs * N_tasks, self.out_ch, H, W)
        out2 = self.conv2d_main2(task_agnostic_out1)
        N_ch, H, W = out2.shape[-3:]

        after_fbn_main2 = []
        for task_input, task_fbn_main2 in zip(out2.reshape(bs, N_tasks, N_ch, H, W).transpose(0, 1), self.fbns_main2):
                after_fbn_main2.append(task_fbn_main2(task_input))
        after_fbn_main2 = torch.stack(after_fbn_main2, dim=1)

        if self.use_opt_path:
            opt_out = self.conv2d_opt(task_agnostic_input)
            N_ch, H, W = opt_out.shape[-3:]

            after_bn_opt = []
            for task_input, task_fbn_opt in zip(opt_out.reshape(bs, N_tasks, N_ch, H, W).transpose(0, 1), self.fbns_opt):
                    after_bn_opt.append(task_fbn_opt(task_input))
            after_bn_opt = torch.stack(after_bn_opt, dim=1)
            after_fbn_main2 += after_bn_opt
        else:
            after_fbn_main2 += x

        after_fbn_main2 = F.relu(after_fbn_main2)

        # apply second mask:
        after_fbn_main2 *= second_conv_masks

        return after_fbn_main2


class BaseGatedConv(nn.Module):
    """
        Base class for single Task-gated conv layer.
        Incorporates all gating logic.
    """
    def __init__(self, in_ch, out_ch, freezing_method,
                 aggregate_firing=False,
                 N_tasks=1,
                 conv_params: dict = None):
        super().__init__()

        self.N_tasks = N_tasks
        self.in_ch = in_ch
        self.out_ch = out_ch

        if conv_params:
            self.conv2d = FreezableConv2d(in_ch, out_ch, **conv_params)
        else:
            self.conv2d = FreezableConv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.main_conv_path = nn.Sequential(self.conv2d)

        self.fbns = nn.ModuleList([create_freezable_bn(self.out_ch) for _ in range(self.N_tasks)])

        self.gates = nn.ModuleList([self.create_gate_fc() for _ in range(self.N_tasks)])

        self.taskwise_sparse_objective = BufferList([torch.empty((1))] * self.N_tasks) # used for sparsity objective

        # aggregates frequencies with which kernels were chosen
        self.aggregate_firing = aggregate_firing
        self.channels_firing_freq = torch.zeros((self.N_tasks, self.out_ch))
        self.n_aggregations = torch.zeros((self.N_tasks))  # for calculating probabilities correctly

        self.freezing_method = freezing_method
        
        self.register_buffer('frozen_kernels_mask',
                             torch.zeros((self.out_ch), dtype=int))

    def create_gate_fc(self):
        gate_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.in_ch, 16),
            nn.BatchNorm1d(16, track_running_stats=False) if cfg.USE_BATCHNORM_GATES else nn.Identity(),
            nn.ReLU(),
            nn.Linear(16, self.out_ch)
        )
        return gate_fc

    def add_task_path(self):
        r"""Add task path to the block"""
        r"""Following .to(some_weight) operation is to ensure, that all the gates and fbns are on the same device
           This method, however, will not work with fbns, if affine=False, so be aware of that."""
        self.gates.append(self.create_gate_fc().to(cfg.DEVICE))
        self.fbns.append(create_freezable_bn(self.out_ch).to(cfg.DEVICE))

        self.taskwise_sparse_objective.append(torch.empty((1)))

        self.channels_firing_freq = torch.cat([self.channels_firing_freq, 
                                               torch.zeros((1, self.out_ch))], 0)
        self.n_aggregations = torch.cat([self.n_aggregations,
                                        torch.zeros((1))], 0)

        self.N_tasks += 1

    def enable_gates_firing_tracking(self):
        self.aggregate_firing = True

    def reset_gates_firing_tracking(self):
        self.aggregate_firing = False
        self.channels_firing_freq = torch.zeros((self.N_tasks, self.out_ch))
        self.n_aggregations = torch.zeros((self.N_tasks))

    def aggregate_channels_firing_stat(self, channels_mask, task_idx):
        """
            Sums up frequencies of choosing kernels among batches
            during validation or test.

            Attributes:
            channels_mask - binary mask
        """
        self.channels_firing_freq[task_idx] += channels_mask.float().mean(dim=0).detach().cpu()
        self.n_aggregations[task_idx] += 1

    def update_relevant_kernels(self, task_id):
        """
            Updates relevant kernels according to each gate-path e.g. task
        """
        if self.freezing_method.freeze_fixed_proc:
            k = int(self.out_ch * self.freezing_method.freeze_top_proc)
            aggregated_times = self.n_aggregations[task_id]
            threshold = self.freezing_method.freeze_prob_thr * aggregated_times

            gate_stat = self.channels_firing_freq[task_id].clone()
            n_relevant = (gate_stat > threshold).long().sum()
            # gate_stat[gate_stat < threshold] = 0

            if n_relevant > k:
                print(f'Not enough capacity for relevant kernels: {n_relevant}/{k} ')
                idx_to_freeze = torch.topk(gate_stat, k, dim=-1)[1]
            else:
                idx_to_freeze = torch.topk(gate_stat, n_relevant, dim=-1)[1]

        else:
            gate_stat = self.channels_firing_freq[task_id]
            aggregated_times = self.n_aggregations[task_id]
            idx_to_freeze = gate_stat > self.freezing_method.freeze_prob_thr * aggregated_times

        # aggregated mask becomes non-binary, but this does not interfere
        # with the logic of self.freeze_relevant_kernels()
        # and underlines the relevances of the kernels once more
        self.frozen_kernels_mask[idx_to_freeze] += 1

    def freeze_relevant_kernels(self, task_id):
#         from pdb import set_trace; set_trace()
        self.update_relevant_kernels(task_id)
        self.conv2d.freeze(self.frozen_kernels_mask.clamp(0, 1))

        r"""During training of task t only self.fbns[t] tracked relevant statistics, 
            therefore it is the only element to freeze"""
        if cfg.NORMALIZATION_IN_BACKBONE:
            self.fbns[task_id].freeze(self.frozen_kernels_mask.clamp(0, 1))


    def reinitialize_irrelevant_kernels(self):
        r"""
        Invoke all freezable classes to reinitialize unfrozen kernels
        """
        self.conv2d.reinit_unfrozen()

        r"""Despite only self.fbns[t] was properly frozen, all fbns should reinit irreevant parameters, 
            according to their is_frozen masks"""
        if cfg.NORMALIZATION_IN_BACKBONE:
            for fbn in self.fbns:
                fbn.reinit_unfrozen()

    def sample_channels_mask(self, logits):
        """
            Samples binary mask to select
            relevant output channel of the convolution

            Attributes:
            logits - logprobabilities of the bernoully variables
                for each output channel of the convolution to be selected
        """
        if self.training:
            if cfg.USE_GUMBEL_SIGMOID:
                channels_mask = gumbel_sigmoid(logits, tau=2/3)
            else:
                bernoully_logits = torch.stack([logits, -logits], dim=0)
                channels_mask = F.gumbel_softmax(bernoully_logits, tau=2/3, hard=True, dim=0)[0]
        else:
            channels_mask = (logits > 0).long()
        return channels_mask

    def select_channels_for_task(self, x, filters, gate_fc, task_idx):
        """
            Performs selection of the output channels for the given task.

            Attributes:
            x - input tensor
            filters - output tensor to be selected from
            gate_fc - sequential model, provides logprobabilities for each output channel of the convolution
            task_idx - int label of the task path; used for gate firing aggregation
        """
#         from pdb import set_trace; set_trace()
        logits = gate_fc(x)
        mask = self.sample_channels_mask(logits)
        self.taskwise_sparse_objective[task_idx] = mask.float().mean().reshape(1)

        if self.aggregate_firing:
            self.aggregate_channels_firing_stat(mask, task_idx)

        # expand last 2 dims for channel-level elementwise multiplication
        mask = mask[:, :, None, None]
        return filters * mask

    def forward(self, x):

        # task-agnostic conv2D over 5D tensor
        bs, N_tasks, N_ch, H, W = x.shape
        filters = self.main_conv_path(x.reshape(bs * N_tasks, N_ch, H, W))
        N_ch, H, W = filters.shape[-3:]
        filters = filters.reshape(bs, N_tasks, N_ch, H, W)

        after_bn = []
        for task_input, task_fbn in zip(filters.transpose(0, 1), self.fbns):
                after_bn.append(F.relu(task_fbn(task_input)))

        # permute batch_size and N_tasks dims for task-wise iterations
        x = x.transpose(0, 1)
        after_bn = torch.stack(after_bn, dim=0)

        output = []
        for task_idx, (task_input, task_filters, task_gate) in enumerate(zip(x, after_bn, self.gates)):
            selected = self.select_channels_for_task(
                task_input, task_filters, task_gate, task_idx)
            output.append(selected)

        # TODO: change torch.stack(smth).transpose(0, 1) to torch.stack(smth, 1)
        # permute back to batch_size x N_tasks x N_channels x H x W
        output = torch.stack(output).transpose(0, 1)
        # from pdb import set_trace; set_trace()
        return output


class GatedConvDownBlock(BaseGatedConv):
    """
        Gated convolution module for N tasks.
        Assumes data shape of batch_size x N_tasks x N_channels x H x W
        Assumes batch size to be > 1
        Uses maxpool at the end.
    """
    def __init__(self, in_ch, out_ch, freezing_method,
                 aggregate_firing=False,
                 conv_params : dict = None):

        super().__init__(in_ch, out_ch, freezing_method,
                 aggregate_firing=aggregate_firing,
                 conv_params=conv_params)

        self.main_conv_path = nn.Sequential(
                self.conv2d,
                # self.fbn if cfg.USE_BATCHNORM_BACKBONE else nn.Identity(), # Batchnorm cannot be used that easily
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
    

class GatedConvSameBlock(BaseGatedConv):
    """
        Gated convolution module for N tasks.
        Assumes data shape of batch_size x N_tasks x N_channels x H x W
        Assumes batch size to be > 1
        Does not use maxpool at the end.
    """
    def __init__(self, in_ch, out_ch, freezing_method,
                 aggregate_firing=False,
                 conv_params : dict = None):

        super().__init__(in_ch, out_ch, freezing_method,
                 aggregate_firing=aggregate_firing,
                 conv_params=conv_params)


class TaskAgnosticWrapper(nn.Module):
    """
        Layer, that reshapes data to become task-agnostic,
        applies given sequential module, then
        reshapes the data back.

        Useful for applying MaxPool, BN, ReLU, etc outside
        given gated classes.
    """

    def __init__(self, net : nn.Sequential):
        super().__init__()
        self.net = net

    def forward(self, x):
        bs, N_tasks, in_ch, H, W = x.shape
        out = self.net(x.reshape(bs * N_tasks, in_ch, H, W))

        out_ch, H, W = out.shape[-3:]
        out = out.reshape(bs, N_tasks, out_ch, H, W)
        return out
