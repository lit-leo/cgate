r"""Freezable versions of several layers.

This module represents several modified layers, which parameters can be frozen according to the provided mask.

    Typical usage example:
    fconv = FreezableConv2d(3, 5, kernel_size=9)
    fconv.freeze(torch.LongTensor([1, 1, 0, 0, 1]))
    fconv.reinit_unfrozen()
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class _FreezableNormBase(nn.Module):
    """Common base of _InstanceNorm and _BatchNorm"""
    def __init__(self, norm_class_str: str, num_channels: int,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.num_channels = num_channels
        if norm_class_str == 'BatchNorm2d':
            self.norm_class = nn.BatchNorm2d(num_channels, affine, track_running_stats)
        elif norm_class_str == 'InstanceNorm2d':
            self.norm_class = nn.InstanceNorm2d(num_channels, affine, track_running_stats)
        else:
            raise NotImplementedError

        self.register_buffer('mask', torch.zeros(num_channels))
        self.register_buffer('shadow_weight', torch.Tensor(*self.norm_class.weight.shape))
        self.register_buffer('shadow_bias', torch.Tensor(*self.norm_class.bias.shape))
        self.register_buffer('shadow_running_mean', torch.Tensor(*self.norm_class.running_mean.shape))
        self.register_buffer('shadow_running_var', torch.Tensor(*self.norm_class.running_var.shape))
        self.reset_shadow_parameters()

    def reset_shadow_parameters(self):
        self.shadow_weight.data.copy_(self.norm_class.weight)
        self.shadow_bias.data.copy_(self.norm_class.bias)
        self.shadow_running_mean.data.copy_(self.norm_class.running_mean)
        self.shadow_running_var.data.copy_(self.norm_class.running_var)

    def copy_params2shadow_by_idx(self, idx):
        self.shadow_weight.data[idx] = self.norm_class.weight.data[idx].clone()
        self.shadow_bias.data[idx] = self.norm_class.bias.data[idx].clone()
        self.shadow_running_mean.data[idx] = self.norm_class.running_mean.data[idx].clone()
        self.shadow_running_var.data[idx] = self.norm_class.running_var.data[idx].clone()

    def replace_params_with_frozen(self):
        self.norm_class.weight.data = (self.shadow_weight * self.mask +
                                       self.norm_class.weight * (1. - self.mask)).data
        self.norm_class.bias.data = (self.shadow_bias * self.mask +
                                     self.norm_class.bias * (1. - self.mask)).data
        self.norm_class.running_mean.data = (self.shadow_running_mean * self.mask +
                                             self.norm_class.running_mean * (1. - self.mask)).data
        self.norm_class.running_var.data = (self.shadow_running_var * self.mask +
                                            self.norm_class.running_var * (1. - self.mask)).data

    def freeze(self, mask):
        r"""
        Updates is_frozen mask.
        Assumes, that each new mask is based on previous with some
        more elements to be frozen.
        Args:
            mask: torch.Tensor, ones in the mask indicate the kernels to be frozen

        Returns:
            None
        """
        assert mask.ndim == 1
        new_mask = mask.detach().to(self.norm_class.weight)
        diff = (self.mask - new_mask)
        # -1 in diff means that there is a new kernel to be frozen
        new_idx_to_freeze = (diff == -1).nonzero(as_tuple=False)[:, 0].long()
        self.copy_params2shadow_by_idx(new_idx_to_freeze)

        mask = new_mask
        self.register_buffer('mask', mask.contiguous())

    def reinit_unfrozen(self):
        r"""
        Reinitialize currently not frozen parameters.

        shadow_parameters already include copies of the frozen parameters,
        and will be substituted in the proper places
        during replace_params_with_frozen call, so all non-shadow parameters
        can be reinitialized
        Returns:
            None
        """
        self.norm_class.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.replace_params_with_frozen()
        res = self.norm_class(x)
        self.replace_params_with_frozen()
        r"""The self.replace_params_with_frozen is called for the second time to ensure the correct frozen 
            representation of the parameters if they are e.g. accessed from the outside"""
        return res


class FreezableInstanceNorm2d(_FreezableNormBase):
    """
        InstanceNorm2d layer with selectively frozen weight, bias, running_mean and running_var.

        Ones in self.mask indicate, which outputs are frozen.
    """

    def __init__(self, num_channels, affine=True, track_running_stats=True):
        super().__init__('InstanceNorm2d', num_channels, affine, track_running_stats)

    def __repr__(self):
        return f'FreezableInstanceNorm2d; num_channels={self.num_channels}'


class FreezableBatchNorm2d(_FreezableNormBase):
    """
        BatchNorm2d layer with selectively frozen weight, bias, running_mean and running_var.

        Ones in self.mask indicate, which outputs are frozen.
    """

    def __init__(self, num_channels, affine=True, track_running_stats=True):
        super().__init__('BatchNorm2d', num_channels, affine, track_running_stats)

    def __repr__(self):
        return f'FreezableBatchNorm2d; num_channels={self.num_channels}'


class FreezableConv2d(torch.nn.Conv2d):
    """
        Conv2d layer with selectively frozen outputs.

        Ones in self.mask indicate, which outputs are frozen.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias, padding_mode=padding_mode)

        self.register_buffer('mask', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('shadow_weight', torch.Tensor(*self.weight.shape))
        self.register_buffer('shadow_bias', torch.Tensor(*self.bias.shape))

        self.reset_shadow_parameters()

    def reset_shadow_parameters(self):
        self.shadow_weight.data.copy_(self.weight)
        self.shadow_bias.data.copy_(self.bias)

    def copy_weight2shadow_by_idx(self, idx):
        self.shadow_weight.data[idx] = self.weight.data[idx].clone()
        self.shadow_bias.data[idx] = self.bias.data[idx].clone()

    @property
    def frozen_weight(self):
        return self.shadow_weight * self.mask + self.weight * (1. - self.mask)

    @property
    def frozen_bias(self):
        return self.shadow_bias * self.mask.squeeze() + self.bias * (1. - self.mask.squeeze())

    def freeze(self, mask):
        r"""
        Updates is_frozen mask.
        Assumes, that each new mask is based on previous with some
        more elements to be frozen.
        Args:
            mask: torch.Tensor, ones in the mask indicate the kernels to be frozen

        Returns:
            None
        """
        assert mask.ndim == 1
        new_mask = mask.detach().to(self.weight)
        diff = (self.mask[:, 0, 0, 0] - new_mask)
        # -1 in diff means that there is a new kernel to be frozen
        new_idx_to_freeze = (diff == -1).nonzero(as_tuple=False)[:, 0].long()
        self.copy_weight2shadow_by_idx(new_idx_to_freeze)

        mask = new_mask[:, None, None, None]
        self.register_buffer('mask', mask.contiguous())

    def reinit_unfrozen(self):
        r"""
        Reinitialize the self.weight and self.bias.

        shadow_weight and shadow_bias already include frozen weights,
        which will be substituted in the proper places
        during frozen_weight/frozen_bias call, so whole self.weight and self.bias
        can be reinitialized.
        Returns:
            None
        """
        self.reset_parameters()

    def forward(self, x):
        return F.conv2d(x, self.frozen_weight, bias=self.frozen_bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)
