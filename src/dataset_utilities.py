from __future__ import print_function

import copy
import os

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms, datasets

from noise_dataset_class import NoiseDataset

# Normalization for CIFAR10 dataset
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def insert_sample_to_dataset(trainloader, sample_to_insert_data, sample_to_insert_label):
    """
    Inserting test sample into the trainset
    :param trainloader: contains the trainset
    :param sample_to_insert_data: the data which we want to insert
    :param sample_to_insert_label: the data label which we want to insert
    :return: dataloader which contains the trainset with the additional sample
    """
    sample_to_insert_label_expended = np.expand_dims(sample_to_insert_label, 0)
    sample_to_insert_data_expended = np.expand_dims(sample_to_insert_data, 0)

    if isinstance(trainloader.dataset.train_data, torch.Tensor):
        sample_to_insert_data_expended = torch.Tensor(sample_to_insert_data_expended)

    # # Insert sample to train dataset
    dataset_train_with_sample = copy.deepcopy(trainloader.dataset)
    dataset_train_with_sample.train_data = np.concatenate((trainloader.dataset.train_data,
                                                           sample_to_insert_data_expended))
    dataset_train_with_sample.train_labels = np.concatenate((trainloader.dataset.train_labels,
                                                             sample_to_insert_label_expended))

    if isinstance(trainloader.dataset.train_data, torch.Tensor) and \
            not isinstance(dataset_train_with_sample.train_data, torch.Tensor):
        dataset_train_with_sample.train_data = \
            torch.tensor(dataset_train_with_sample.train_data,
                         dtype=trainloader.dataset.train_data.dtype)

    # Create new dataloader
    trainloader_with_sample = data.DataLoader(dataset_train_with_sample,
                                              batch_size=trainloader.batch_size,
                                              shuffle=True,
                                              num_workers=trainloader.num_workers)
    return trainloader_with_sample


def create_svhn_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for SVHN dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """

    trainset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              normalize]))
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    data_dir = os.path.join(data_dir, 'svhn')
    testset = datasets.SVHN(root=data_dir,
                            split='test',
                            download=True,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          normalize]))

    # Align as CIFAR10 dataset
    testset.test_data = testset.data
    testset.test_labels = testset.labels

    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

    # Classes name
    classes_cifar10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes_svhn = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

    return trainloader, testloader, classes_svhn, classes_cifar10


def create_cifar10_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for CIFAR10 dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    trainset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              normalize]))
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    testset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               download=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             normalize]))
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def create_cifar100_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for CIFAR100 dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    trainset = datasets.CIFAR100(root=data_dir,
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([transforms.ToTensor(),
                                                               normalize]))
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    testset = datasets.CIFAR100(root=data_dir,
                                train=False,
                                download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              normalize]))
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
               'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
               'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
               'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
               'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
               'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
               'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
               'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
               'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
               'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
               'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
               'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
               'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
               'worm')

    return trainloader, testloader, classes


def generate_noise_sample():
    random_sample_data = np.random.randint(256, size=(32, 32, 3), dtype='uint8')
    random_sample_label = -1
    return random_sample_data, random_sample_label


class CIFAR10RandomLabels(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.

    Params
    ------
    corrupt_prob: float
        Default 0.0. The probability of a label being replaced with
        random label.
    num_classes: int
        Default 10. The number of classes in the dataset.
    """

    def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.train_labels if self.train else self.test_labels)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        if self.train:
            self.train_labels = labels
        else:
            self.test_labels = labels


def create_cifar10_random_label_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4,
                                            label_corrupt_prob=1.0):
    """
    create train and test pytorch dataloaders for CIFAR10 dataset.
    Train set can be with random labels, the probability to be random depends on label_corrupt_prob.
    :param data_dir: the folder that will contain the data.
    :param batch_size: the size of the batch for test and train loaders.
    :param label_corrupt_prob: the probability to be random of label of train sample.
    :param num_workers: number of cpu workers which loads the GPU with the dataset.
    :return: train and test loaders along with mapping between labels and class names.
    """

    # Trainset with random labels
    trainset = CIFAR10RandomLabels(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                 normalize]),
                                   corrupt_prob=label_corrupt_prob)
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    # Testset with real labels
    testset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               download=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             normalize]))
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def create_cifar10_dataloaders_with_training_subset(data_dir: str = './data', batch_size: int = 128,
                                                    num_workers: int = 4,
                                                    trainset_size: int = 50000,
                                                    adjust_sgd_update=False):
    """
    create train and test pytorch dataloaders for CIFAR10 dataset.
    Train set is a subset of the orignal CIFAR10 dataset.
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders.
    :param trainset_size: the number of samples of the trainset
    :param adjust_sgd_update: weather to upsample the trainset (make duplicate) so the total size will be as the original.
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    np.random.seed(12345)
    trainset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              normalize]))
    trainset_size_org = len(trainset)

    # Get trainset subset
    trainset.train_data = trainset.train_data[:trainset_size]
    trainset.train_labels = trainset.train_labels[:trainset_size]

    if adjust_sgd_update is True:
        if 50000 % trainset_size != 0:
            raise NameError('trainset_size_org / trainset_size is not a int')
        duplicate_trainset_num = int(trainset_size_org / trainset_size)
        trainset.train_data = np.repeat(trainset.train_data, duplicate_trainset_num, axis=0)
        trainset.train_labels = np.repeat(trainset.train_labels, duplicate_trainset_num, axis=0)

    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    testset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               download=True,
                               transform=transforms.Compose([transforms.ToTensor(),
                                                             normalize]))
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def create_mnist_dataloaders(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for MNIST dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """

    # Normalization for MNIST dataset
    normalize_mist = transforms.Normalize(mean=[0.1307], std=[0.3081])
    trainset = datasets.MNIST(root=data_dir,
                              train=True,
                              download=True,
                              transform=transforms.Compose([transforms.ToTensor(),
                                                            normalize_mist]))
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    testset = datasets.MNIST(root=data_dir,
                             train=False,
                             download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           normalize_mist]))
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    return trainloader, testloader, classes


def dataloaders_noise(data_dir: str = './data', batch_size: int = 128, num_workers: int = 4):
    """
    create train and test pytorch dataloaders for CIFAR10 dataset
    :param data_dir: the folder that will contain the data
    :param batch_size: the size of the batch for test and train loaders
    :param num_workers: number of cpu workers which loads the GPU with the dataset
    :return: train and test loaders along with mapping between labels and class names
    """
    trainset = datasets.CIFAR10(root=data_dir,
                                train=True,
                                download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              normalize]))
    trainloader = data.DataLoader(trainset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    testset = NoiseDataset(root=data_dir,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         normalize]))
    testloader = data.DataLoader(testset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataloaders = {'train': trainloader,
                   'test': testloader,
                   'classes': classes,
                   'classes_noise': ('Noise',) * 10}
    return dataloaders
