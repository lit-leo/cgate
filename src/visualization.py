r"""Visualization instruments

Used in conjunction with jupyter notebook to check sparsities of the gated layers
    and track, how much capacity is frozen by each task

    Typical usage example:
    visualize_sparsity('results/experiment_tag')
"""
import os
import pickle

import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from iteround import saferound

from config import cfg
from src.train import aggregate_firing_stat_on_data


def get_gate_sparsity_data_dependent(litmodel, data_loader, gate_path_num, verbose=False):
    r"""
    Display gates' frequencies of choosing kernels based on some data
    Args:
        litmodel: ChannelGatedCL
        data_loader: torch.utils.data.DataLoader
        gate_path_num: the number of task, which gates are going to be examined
        verbose: bool, specifying use of tqdm

    Returns:
        firing stat for all layers for a particular task
    """
    stat = aggregate_firing_stat_on_data(litmodel, data_loader, verbose)

    # select particular gate statistics
    gate_stat = [(freq[gate_path_num] / n_agg[gate_path_num]).unsqueeze(0) for (freq, n_agg) in stat]
    return gate_stat


def visualize_gate_sparsity_data_dependent(litmodel, data_loader, gate_path_num, verbose=False, sort_channels=False):
    r"""
    Plot a heatmap of gates' frequencies of choosing kernels based on some data
    """
    gate_stat = get_gate_sparsity_data_dependent(litmodel, data_loader, gate_path_num, verbose=verbose)
    plot_stat_heatmaps_layerwise(gate_stat, sort=sort_channels)


def extract_taskwise_stat(rootdir):
    r"""
    Extract saved firing frequencies from the rootdir
    Args:
        rootdir: str, path to the experiment folder,
            usually something like 'results/expetiment_tag'

    Returns:
        Task-ordered firing frequencies
    """
    pickle_fnames = [fname for fname in os.listdir(rootdir) if '.pickle' in fname]
    task_sorted_fnames = sorted(pickle_fnames)
    
    gates_stat = []
    for task_idx, fname in enumerate(task_sorted_fnames):
        with open(os.path.join(rootdir, fname), 'rb') as f:
            data = pickle.load(f)
        task_stat = [freq[task_idx] / n_agg[task_idx] for (freq, n_agg) in data['gates_freq']]
        gates_stat.append(task_stat)
    return gates_stat


def convert_to_layerwise_stat(taskwise_stat):
    r"""
    Converd task-order to layer-order
    Args:
        taskwise_stat: Task-ordered firing frequencies

    Returns:
        Layer-ordered firing frequencies
    """
    n_tasks = len(taskwise_stat)
    n_layers = len(taskwise_stat[0])
    layerwise_stat = [[] for _ in range(n_layers)]
    for task_stat in taskwise_stat:
        for i, layer_stat in enumerate(task_stat):
            layerwise_stat[i].append(layer_stat)

    layerwise_stat = [torch.stack(layer_stat) for layer_stat in layerwise_stat]
    return layerwise_stat


def extract_taskwise_frozen_masks(rootdir):
    r"""
    Extract task-ordered frozen kernels masks
    Args:
        rootdir: str, path to the experiment folder,
            usually something like 'results/expetiment_tag'

    Returns:
        Task-ordered frozen kernels masks
    """
    pickle_fnames = [fname for fname in os.listdir(rootdir) if '.pickle' in fname]
    task_sorted_fnames = sorted(pickle_fnames)

    taskwise_masks = []
    for task_idx, fname in enumerate(task_sorted_fnames):
        with open(os.path.join(rootdir, fname), 'rb') as f:
            data = pickle.load(f)
        taskwise_masks.append(data['frozen_kernels_mask'])

    return taskwise_masks


def extract_layerwise_frozen_masks(rootdir):
    r"""
    Extract layer-ordered frozen kernels masks
    Args:
        rootdir: str, path to the experiment folder,
            usually something like 'results/expetiment_tag'

    Returns:
        Layer-ordered frozen kernels masks
    """
    taskwise_masks = extract_taskwise_frozen_masks(rootdir)
    n_tasks = len(taskwise_masks)
    n_layers = len(taskwise_masks[0])
    layerwise_masks = [[] for _ in range(n_layers)]
    for task_mask in taskwise_masks:
        for i, layer_mask in enumerate(task_mask):
            layerwise_masks[i].append(layer_mask)

    layerwise_masks = [torch.stack(layer_mask) for layer_mask in layerwise_masks]
    return layerwise_masks


def convert_count_frozen_to_diff_frozen(layerwise_masks):
    r"""
    Represent frozen masks in a fashion,
        where task is mapped to a mask of kernels, frozen solely by it

    By default all extracted frozen kernels information are presented in a cumulative manner:
        on task k we see all kenrels, which had been frozen during tasks 0..k inclusive.
    Args:
        layerwise_masks: layer-ordered frozen kernels masks

    Returns:
        Layer-ordered frozen kernels masks, frozen by individual tasks
    """
    for i, layer_mask in enumerate(layerwise_masks):
        layer_mask = layer_mask.clamp(0, 1)
        layer_mask_rolled = layer_mask.roll(1, 0)
        layer_mask_rolled[0].zero_()
        layerwise_masks[i] = layer_mask - layer_mask_rolled
    return layerwise_masks


def convert_count_frozen_to_n_frozen(layerwise_masks):
    r"""
    Count, how many kernels were frozen by each task
    Args:
        layerwise_masks: layer-ordered frozen kernels masks

    Returns:
        list, containing ints, each representing the number of frozen kernels by each task
    """
    diff_frozen = convert_count_frozen_to_diff_frozen(layerwise_masks)
    diff_frozen = [i.nonzero(as_tuple=True)[0].unique(return_counts=True) for i in diff_frozen]
    return diff_frozen


def plot_stat_heatmaps_layerwise(layerwise_stat, sort=True):
    r"""
    Plot a heatmap, displaying kernels' usage frequency
    Args:
        layerwise_stat: layer-ordered firing frequencies
        sort: bool, if True - sort all kernels by the their usage frequency among all tasks
            thus reused kernels appear rightmost

    Returns:
        None
    """
    fig, axs = plt.subplots(len(layerwise_stat), 1, figsize=(10, len(layerwise_stat) * 3))
    for i, (layer_stat, ax) in enumerate(zip(layerwise_stat, axs)):
        if sort:
            viz_stat = layer_stat[:, np.argsort(-layer_stat.sum(dim=0))]
        else:
            viz_stat = layer_stat
        sns.heatmap(viz_stat, ax=ax, cmap='viridis')
        ax.set_title(f'Layer {i} kernels usage frequency on validation')
        ax.set_ylabel('Task idx')
        ax.set_xlabel('Kernel idx')
    fig.tight_layout()
    plt.show()


def visualize_frozen_kernels_count(frozen_kernels_count):
    r"""
    Plot bar diagram, representing, how much capacity was frozen by each task in each layer
    Args:
        frozen_kernels_count: list, containing how much kernels were frozen by each task

    Returns:
        None
    """
    frozen_kernels = [i[1].tolist() + [0] * (5 - len(i[1])) for i in frozen_kernels_count]
    if len(frozen_kernels) == 3:
        # SimpleCNN case
        total_capacity = [cfg.CONV_CH, cfg.CONV_CH, cfg.CONV_CH]
    elif len(frozen_kernels) == 9:
        # resnet18 case
        total_capacity = [64, 64, 64, 128, 128, 256, 256, 512, 512]
    else:
        raise NotImplementedError
        
    frozen_kernels = [entry + [total - sum(entry)] for (entry, total) in zip(frozen_kernels, total_capacity)]
    layerwise_frozen_kernels_dict = {f'Layer {i}': v for i, v in enumerate(frozen_kernels)}
    category_names = ['Task_0', 'Task_1', 'Task_2', 'Task_3', 'Task_4', 'Unused']
    survey(layerwise_frozen_kernels_dict, category_names, total_capacity)
    plt.show()


def survey(results, category_names, total_capacity):
    """
    Courtesy of https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()), dtype=float)

    # normalize the data
    data = data / np.array(total_capacity)[:, None] * 100
    data = np.array([saferound(row, 0) for row in data]).astype(int)

    data_cum = data.cumsum(axis=1)
    category_colors = list(plt.get_cmap('RdYlGn')(
        np.linspace(0.2, 0.8, data.shape[1] - 1))) + ['lightgray']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        text_color = 'black'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            if c > 2: # if space is sufficient to print percentages
                ax.text(x, y, f'{c}%', ha='center', va='center',
                        color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize=12)
    ax.set_ylabel('Percent of kernels frozen by each task')
    return fig, ax


def visualize_sparsity(rootdir):
    r"""
    Function that plots kernels usage heatmap
        and returns frozen kernels statistics
        to be shown in the Jupyter notebook
    Args:
        rootdir: str, path to the experiment folder,
            usually something like 'results/expetiment_tag'

    Returns:
        None
    """
    kernels_frozen = extract_layerwise_frozen_masks(rootdir)
    layerwise_stat = convert_to_layerwise_stat(extract_taskwise_stat(rootdir))
    plot_stat_heatmaps_layerwise(layerwise_stat)
    visualize_frozen_kernels_count(convert_count_frozen_to_n_frozen(kernels_frozen))


def plot_gates_hist_for_task(stat, task_idx):
    r"""
    Deprecated function, that was used to assess the gates' sparsities in
        SimpleCNN architecture.
    Args:
        stat: list of tuples (aggregated_stat, n_aggregations)
        task_idx: int

    Returns:
        None
    """
    plt.figure(figsize=(10, 12))
    
    plt.subplot(311)
    plt.title('Gates firing histogram for the first layer')
    gates, norm_const = stat[0]
    plt.hist(gates[task_idx] / norm_const[task_idx], bins=20)

    plt.subplot(312)
    plt.title('Gates firing histogram for the second layer')
    gates, norm_const = stat[1]
    plt.hist(gates[task_idx] / norm_const[task_idx], bins=20)

    plt.subplot(313)
    plt.title('Gates firing histogram for the third layer')
    gates, norm_const = stat[2]
    plt.hist(gates[task_idx] / norm_const[task_idx], bins=20)
    
    
def plot_gates_bars_for_task(stat, task_idx):
    r"""
    Deprecated function, that was used to assess the gates' sparsities in
        SimpleCNN architecture.
    Args:
        stat: list of tuples (aggregated_stat, n_aggregations)
        task_idx: int

    Returns:
        None
    """
    plt.figure(figsize=(10, 12))
    
    plt.subplot(311)
    plt.title('Gates frequencies for the first layer')
    gates, norm_const = stat[0]
    plt.bar(np.arange(gates.shape[1]), gates[task_idx] / norm_const[task_idx])

    plt.subplot(312)
    plt.title('Gates frequencies for the second layer')
    gates, norm_const = stat[1]
    plt.bar(np.arange(gates.shape[1]), gates[task_idx] / norm_const[task_idx])

    plt.subplot(313)
    plt.title('Gates frequencies for the third layer')
    gates, norm_const = stat[2]
    plt.bar(np.arange(gates.shape[1]), gates[task_idx] / norm_const[task_idx])