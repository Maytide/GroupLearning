from copy import deepcopy
from functools import reduce

import numpy as np
import torch
import torch.nn.functional as F
import torchvision


class MNISTDataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.data[idx, :].float()
        targets = self.targets[idx, :].float()
        if self.transform is not None:
            data = self.transform(data)

        return data, targets


class Reshape(object):

    def __init__(self):
        return

    def __call__(self, data):
        # k = reduce(lambda a, b: a*b, data.shape[1:])
        # data = data.view(data.shape[0], k)
        # print('shape k', data.shape, k, len(data.shape))
        if len(data.shape) == 2:
            data = data.view(data.shape[0] * data.shape[1])
        elif len(data.shape) > 2:
            k = reduce(lambda a, b: a * b, data.shape[1:])
            data = data.view(data.shape[0], k)

        return data


class Scale(object):

    def __init__(self):
        return

    def __call__(self, data):
        return data.float() / 255


def load_mnist(n, savedir='./datasets'):
    mnist_trainset = torchvision.datasets.MNIST(root=savedir, train=True, download=True, transform=None)
    mnist_testset = torchvision.datasets.MNIST(root=savedir, train=False, download=True, transform=None)

    data_train = mnist_trainset.data[:n]
    targets_train = mnist_trainset.targets[:n]
    data_test = mnist_testset.data[:n]
    targets_test = mnist_testset.targets[:n]
    targets_train = labels_to_oneHot(targets_train, 10)
    targets_test = labels_to_oneHot(targets_test, 10)

    return data_train, targets_train, data_test, targets_test


def corrupt_labels(targets, num_corrupt, savedir='./datasets'):
    assert 0 <= num_corrupt <= targets.shape[1]
    corrupt_targets = deepcopy(targets)

    labels = oneHot_to_labels(targets)
    for corrupt_target, label in zip(corrupt_targets, labels):
        # [label,] = oneHot_to_labels(target[None])
        others = list(range(0, label)) + list(range(label+1, 10))
        r = np.random.choice(others, num_corrupt, replace=False)
        corrupt_target[r] = 1

    return corrupt_targets


def oneHot_to_labels(oneHot):
    return np.argmax(oneHot, axis=1)


def labels_to_oneHot(targets, dim):
    oneHot = np.zeros((targets.shape[0], dim))
    oneHot[np.arange(targets.shape[0]), targets] = 1

    return torch.from_numpy(oneHot)