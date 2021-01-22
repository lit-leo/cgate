"""Tasked dataset for multitask continual learning setup.

Module contains TaskedDataset class, which controls its content based
on the current task_id, which can be set by set_task() function.

prepare_dataset() function fetches the data according to the
dataset name provided, splits train into train-validation and
constructs appropriate TaskedDataset instances.

get_loaders() return loaders for the passed datasets.

set_task() is used to control the current task in the train, val and
test data simultaneously.


    Typical usage example:

    datasets = prepare_dataset(dataset_name='MNIST', truncate_size=None,
                               train_transform=None, val_transform=None,
                               test_transform=None)
    dataloaders = get_loaders(*datasets, batch_size=256)
    train_data, val_data, test_data = datasets
    train_loader, val_loader, test_loader = dataloaders
    task_num = 0
    set_task(task_num, train_data, val_data, test_data)
"""
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from sklearn.model_selection import train_test_split


class _CPUDataset(Dataset):
    """
        A small syntetic dataset for
        debugging model on cpu.
    """

    def __init__(self, in_ch=1, height=28):
        super().__init__()
        self.data = torch.randn((10, in_ch, height, height), dtype=torch.float)
        self.targets = torch.ones((10), dtype=torch.long)
        self.current_task = None

    def set_task(self, task_id):
        self.current_task = task_id

    def get_random_coreset(self, size=30):
        """
            Placeholder for back compatability with
            other modules.
        """
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.current_task


class Coreset(Dataset):
    def __init__(self, coreset_data, transforms=None):
        super().__init__()
        self.transform = transforms
        self.data = coreset_data

    def populate(self, coreset_data):
        self.data = coreset_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target, task_id = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)

        # add channel dimention
        img = img[None, ...]
        return img, target, task_id


class TaskedDataset(Dataset):
    """ Custom dataset class, which items are controlled by the task label"""
    def __init__(self, X : torch.FloatTensor, y : torch.LongTensor, transform=None,
                 all_tasks : list =[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
                 truncate_size : int =None):
        super().__init__()
        self.all_tasks = all_tasks
        self.current_task = None
        self.truncate_size = truncate_size

        assert isinstance(X, torch.FloatTensor)
        assert isinstance(y, torch.LongTensor)
        self.all_y = y.clone()
        self.all_x = X.clone()
        self.y = self.all_y
        self.X = self.all_x
        self.transform = transform

    def __getitem__(self, idx):
        img = self.X[idx]
        if self.transform is not None:
            img = self.transform(img)

        # add channel dimension if needed
        if img.ndim == 2:
            img = img[None, ...]
        return img, self.y[idx], self.current_task

    def __len__(self):
        return self.y.shape[0]

    def binarise(self, target):
        y1, y2 = target.unique()
        target[target == y1] = 0
        target[target == y2] = 1
        return target

    def set_task(self, task_id):
        self.current_task = task_id
        idx = np.isin(self.all_y, self.all_tasks[task_id])
        self.X = self.all_x[idx]
        self.y = self.binarise(self.all_y[idx])
        if self.truncate_size:
            self.X = self.X[:self.truncate_size]
            self.y = self.y[:self.truncate_size]

    def get_random_coreset(self, size=30):
        if size > 0 and self.current_task:  # task is not None or 0
            # select random images from previous tasks
            core_x = []
            core_y = []
            core_task_idx = []
            for i in range(self.current_task):
                tsk_id = np.isin(self.all_y, self.all_tasks[i])
                core_id = np.random.randint(0, np.sum(tsk_id), size)
                core_x.append(self.all_x[tsk_id][core_id])
                core_y.append(self.all_y[tsk_id][core_id])
                core_task_idx.append(torch.LongTensor([i] * size))

            core_x = torch.cat(core_x, dim=0)
            core_y = torch.cat(core_y, dim=0)
            core_task_idx = torch.cat(core_task_idx, dim=0)
            coreset_data = []
            for x, y, t in zip(core_x, core_y, core_task_idx):
                coreset_data.append((x, y, t))

#             print('Coreset for {} class(es).'.format(len(core_x) // size))
#             print('{} images in coreset.'.format(len(coreset_data)))
            return Coreset(coreset_data)
        else:
            return None


def prepare_dataset(dataset_name='MNIST', data_path='data', val_size=0.1,
                    task_pairs=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
                    truncate_size=None,
                    train_transform=None,
                    val_transform=None,
                    test_transform=None):
    r"""
    Fetches the data according to the dataset name provided,
    splits train into train-validation and constructs appropriate TaskedDataset instances.
    Args:
        dataset_name: str, specifying the dataset
        data_path: str, path to store the data
        val_size: float, ration of validation from the all data
        task_pairs: list, which classes are paired up. Numbers represent indexes in intorchvision_dataset.targets field.
        truncate_size: int, used to reduce the size of the dataset
        train_transform: torchvision.transforms
        val_transform: torchvision.transforms
        test_transform: torchvision.transforms

    Returns:
        Three TaskedDataset instances for train, validation and test.
    """
    if dataset_name == 'MNIST':
        dataset_class = datasets.MNIST
    elif dataset_name == 'Fashion-MNIST':
        dataset_class = datasets.FashionMNIST
    elif dataset_name == 'KMNIST':
        dataset_class = datasets.KMNIST
    elif dataset_name == 'CIFAR10':
        dataset_class = datasets.CIFAR10
    elif dataset_name == '_CPU_224':  # for imitating 224x224 data
        tasked_train = _CPUDataset(in_ch=3, height=224)
        tasked_val = _CPUDataset(in_ch=3, height=224)
        tasked_test = _CPUDataset(in_ch=3, height=224)
        return tasked_train, tasked_val, tasked_test
    else:
        raise NotImplementedError

    data_train = dataset_class(root=data_path, train=True,
                               download=True)

    X, y = data_train.data / 255., data_train.targets
    if dataset_name == 'CIFAR10':
        X = torch.FloatTensor(X).permute(0, 3, 1, 2)
        y = torch.LongTensor(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=42)

    tasked_train = TaskedDataset(X_train, y_train,
                                 all_tasks=task_pairs,
                                 truncate_size=truncate_size,
                                 transform=train_transform)
    tasked_val = TaskedDataset(X_val, y_val,
                               all_tasks=task_pairs,
                               truncate_size=truncate_size,
                               transform=val_transform)

    data_test = dataset_class(root=data_path, train=False,
                              download=True)
    X_test, y_test = data_test.data / 255., data_test.targets
    if dataset_name == 'CIFAR10':
        X_test = torch.FloatTensor(X_test).permute(0, 3, 1, 2)
        y_test = torch.LongTensor(y_test)

    tasked_test = TaskedDataset(X_test, y_test,
                                all_tasks=task_pairs,
                                truncate_size=truncate_size,
                                transform=test_transform)

    return tasked_train, tasked_val, tasked_test


def get_loaders(train_dataset, val_dataset, test_dataset, batch_size=128):
    r"""
    Return loaders for the passed datasets
    Args:
        train_dataset: torch.utils.data.Dataset
        val_dataset: torch.utils.data.Dataset
        test_dataset: torch.utils.data.Dataset
        batch_size: int

    Returns:
        Three dataloaders for train, validation and test.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8,
                              shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8,
                            shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8,
                             shuffle=False)

    return train_loader, val_loader, test_loader


def set_task(task_id, train_dataset, val_dataset, test_dataset):
    r"""
    Control the current task in the train, val and test data simultaneously
    Args:
        task_id: int, the id of the task to retrieve data from
        train_dataset: TaskedDataset
        val_dataset: TaskedDataset
        test_dataset: TaskedDataset

    Returns:
        None
    """
    assert isinstance(task_id, int)
    train_dataset.set_task(task_id)
    val_dataset.set_task(task_id)
    test_dataset.set_task(task_id)


def get_coreset_loader(train_dataset, coreset_size=100):
    r"""
    Get dataloader for coreset.
    Used in class-incremental setup.
    Currently unfinished.
    Args:
        train_dataset: TaskedDataset
        coreset_size: int

    Returns:

    """
    coreset = train_dataset.get_random_coreset(size=coreset_size)

    if coreset is not None:
        coreset_loader = DataLoader(coreset, batch_size=128, num_workers=4,
                                    shuffle=True)
    else:
        coreset_loader = None

    return coreset_loader
