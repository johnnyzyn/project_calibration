"""
Create train, valid, test iterators for CIFAR-10.
Train set size: 45000
Val set size: 5000
Test set size: 10000
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler



# def get_train_valid_loader(batch_size,
#                            augment,
#                            random_seed=0,
#                            valid_size=0.1,
#                            shuffle=False,
#                            num_workers=8,
#                            pin_memory=False,
#                            get_val_temp=0,
#                            sub_rand=True,
#                            data_dir='./datasets'):
#     """
#     Utility function for loading and returning train and valid
#     multi-process iterators over the CIFAR-10 dataset.
#     Params:
#     ------
#     - batch_size: how many samples per batch to load.
#     - augment: whether to apply the data augmentation scheme
#       mentioned in the paper. Only applied on the train split.
#     - random_seed: fix seed for reproducibility.
#     - valid_size: percentage split of the training set used for
#       the validation set. Should be a float in the range [0, 1].
#     - shuffle: whether to shuffle the train/validation indices.
#     - num_workers: number of subprocesses to use when loading the dataset.
#     - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
#       True if using GPU.
#     - get_val_temp: set to 1 if temperature is to be set on a separate
#                     val set other than normal val set.
#     Returns
#     -------
#     - train_loader: training set iterator.
#     - valid_loader: validation set iterator.
#     """
#     error_msg = "[!] valid_size should be in the range [0, 1]."
#     assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

#     mean = [x / 255 for x in [125.3, 123.0, 113.9]]
#     std = [x / 255 for x in [63.0, 62.1, 66.7]]

#     # Data Argumentation
#     lists = [
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(32, padding=4),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ]
#     train_transform = transforms.Compose(lists)
#     valid_transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize(mean, std)]
#     )


#     # load the dataset
#     train_dataset = datasets.CIFAR10(
#         root=data_dir, train=True,
#         download=True, transform=train_transform,
#     )

#     valid_dataset = datasets.CIFAR10(
#         root=data_dir, train=True,
#         download=False, transform=valid_transform,
#     )


#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#     )
#     valid_loader = torch.utils.data.DataLoader(
#             valid_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             pin_memory=True,
#         )

#     return (train_loader, valid_loader)

def get_train_valid_loader(batch_size,
                           augment,
                           random_seed=1,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False,
                           get_val_temp=0,
                           data_dir='./datasets'):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. 
    Params:
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - get_val_temp: set to 1 if temperature is to be set on a separate
                    val set other than normal val set.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=False, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    if get_val_temp > 0:
        valid_temp_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=False, transform=valid_transform,
        )
        split = int(np.floor(get_val_temp * split))
        valid_idx, valid_temp_idx = valid_idx[split:], valid_idx[:split]
        valid_temp_sampler = SubsetRandomSampler(valid_temp_idx)
        valid_temp_loader = torch.utils.data.DataLoader(
            valid_temp_dataset, batch_size=batch_size, sampler=valid_temp_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # print("Train indices:", train_sampler.indices)
    # print("Valid indices:", valid_sampler.indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    if get_val_temp > 0:
        return (train_loader, valid_loader, valid_temp_loader)
    else:
        return (train_loader, valid_loader)


def get_test_loader(batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False,
                    data_dir='./datasets'):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

def get_val_test_loader_from_test_set(batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=False,
                                    valid_size=0.5,
                                    data_dir='./datasets'):

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(1)
        np.random.shuffle(indices)

    valid_idx, test_idx = indices[split:], indices[:split]

    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return valid_loader, test_loader

def get_test_valid_loader(batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False,
                           data_dir='./datasets'):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-100 dataset.
    Params:
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - get_val_temp: set to 1 if temperature is to be set on a separate
                    val set other than normal val set.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    #transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    # load the dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    num_test = len(test_dataset)
    indices = list(range(num_test))
    split = int(np.floor(valid_size * num_test))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    test_idx, valid_idx = indices[split:], indices[:split]
    

    test_sampler = SubsetRandomSampler(test_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return test_loader, valid_loader