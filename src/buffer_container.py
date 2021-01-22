'''Containers for generic tensors mimicking corresponding Parameter containers

see torch.nn.ParameterDict and torch.nn.ParameterList for reference.
'''

import torch
import operator

from collections import OrderedDict
from torch._six import container_abcs

from torch.nn import Module


class BufferList(Module):
    r"""Holds buffers in a list.

    :class:`~torch.nn.BufferList` can be indexed like a regular Python
    list, but buffers it contains are properly registered, and will be
    visible by all :class:`~torch.nn.Module` methods.

    Arguments:
        parameters (iterable, optional): an iterable of :class:`~torch.Tensor` to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.buffers = torch.TensorList([
                    torch.Tensor(torch.randn(10, 10)) for i in range(10)
                ])

            def forward(self, x):
                # BufferList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.buffers):
                    x = self.buffers[i // 2].mm(x) + p.mm(x)
                return x
    """

    def __init__(self, tensors=None):
        super().__init__()
        if tensors is not None:
            self += tensors

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return type(self)(list(self._buffers.values())[idx])
        else:
            idx = self._get_abs_string_index(idx)
            return self._buffers[str(idx)]

    def __setitem__(self, idx, tensor):
        idx = self._get_abs_string_index(idx)
        return self.register_buffer(str(idx), tensor)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

    def __iadd__(self, tensors):
        return self.extend(tensors)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, tensor):
        """Appends a given tensor at the end of the list.

        Arguments:
            tensor (torch.Tensor): buffer to append
        """
        self.register_buffer(str(len(self)), tensor)
        return self

    def extend(self, tensors):
        """Appends tensors from a Python iterable to the end of the list.

        Arguments:
            tensors (iterable): iterable of buffers to append
        """
        if not isinstance(tensors, container_abcs.Iterable):
            raise TypeError("BufferList.extend should be called with an "
                            "iterable, but got " + type(tensors).__name__)
        offset = len(self)
        for i, tensor in enumerate(tensors):
            self.register_buffer(str(offset + i), tensor)
        return self

    def extra_repr(self):
        child_lines = []
        for k, t in self._buffers.items():
            size_str = 'x'.join(str(size) for size in t.size())
            device_str = '' if not t.is_cuda else ' (GPU {})'.format(t.get_device())
            parastr = '{} of size {}{}'.format(
                torch.typename(t), size_str, device_str)
            child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('BufferList should not be called.')


class BufferDict(Module):
    r"""Holds buffers in a dictionary.

    BufferDict can be indexed like a regular Python dictionary, but buffers it
    contains are properly registered, and will be visible by all Module methods.

    :class:`~torch.nn.BufferDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~torch.nn.BufferDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~torch.nn.BufferDict` (the argument to
      :meth:`~torch.nn.BufferDict.update`).

    Note that :meth:`~torch.nn.BufferDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        buffers (iterable, optional): a mapping (dictionary) of
            (string : :class:`~torch.Tensor`) or an iterable of key-value pairs
            of type (string, :class:`~torch.Tensor`)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.buffers = nn.BufferDict({
                        'left': torch.randn(5, 10),
                        'right': torch.randn(5, 10)
                })

            def forward(self, x, choice):
                x = self.buffers[choice].mm(x)
                return x
    """

    def __init__(self, tensors=None):
        super().__init__()
        if tensors is not None:
            self.update(tensors)

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, parameter):
        self.register_buffer(key, parameter)

    def __delitem__(self, key):
        del self._buffers[key]

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.keys())

    def __contains__(self, key):
        return key in self._buffers

    def clear(self):
        """Remove all items from the BufferDict.
        """
        self._buffers.clear()

    def pop(self, key):
        r"""Remove key from the BufferDict and return its parameter.

        Arguments:
            key (string): key to pop from the BufferDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the BufferDict keys.
        """
        return self._buffers.keys()

    def items(self):
        r"""Return an iterable of the BufferDict key/value pairs.
        """
        return self._buffers.items()

    def values(self):
        r"""Return an iterable of the BufferDict values.
        """
        return self._buffers.values()

    def update(self, tensors):
        r"""Update the :class:`~torch.nn.BufferDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`buffers` is an ``OrderedDict``, a :class:`~torch.nn.BufferDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            parameters (iterable): a mapping (dictionary) from string to
                :class:`~torch.Tensor`, or an iterable of
                key-value pairs of type (string, :class:`~torch.Tensor`)
        """
        if not isinstance(tensors, container_abcs.Iterable):
            raise TypeError("BufferDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(tensors).__name__)

        if isinstance(tensors, container_abcs.Mapping):
            if isinstance(tensors, (OrderedDict, BufferDict)):
                for key, tensor in tensors.items():
                    self[key] = tensor
            else:
                for key, tensor in sorted(tensors.items()):
                    self[key] = tensor
        else:
            for j, t in enumerate(tensors):
                if not isinstance(t, container_abcs.Iterable):
                    raise TypeError("BufferDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(t).__name__)
                if not len(t) == 2:
                    raise ValueError("BufferDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(t)) +
                                     "; 2 is required")
                self[t[0]] = t[1]

    def extra_repr(self):
        child_lines = []
        for k, t in self._buffers.items():
            size_str = 'x'.join(str(size) for size in t.size())
            device_str = '' if not t.is_cuda else ' (GPU {})'.format(t.get_device())
            parastr = '{} of size {}{}'.format(
                torch.typename(t), size_str, device_str)
            child_lines.append('  (' + k + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('BufferDict should not be called.')
