import logging

import torch
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100


def get_cifar10_datasets(batch_size: int, root: str) -> tuple[DataLoader, DataLoader]:
    logging.info('Prepare CIFAR-10 datasets...')

    preprocess = tf.Compose([
        tf.ToTensor(),
        tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    augmentation = tf.Compose([
        tf.RandomHorizontalFlip(),
        tf.RandomCrop(32, padding=4),
        preprocess,
    ])

    train_data = CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=augmentation
    )

    test_data = CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=preprocess
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

def get_cifar100_datasets(batch_size: int, root: str) -> tuple[DataLoader, DataLoader]:
    logging.info('Prepare CIFAR-100 datasets...')
    
    preprocess = tf.Compose([
        tf.ToTensor(),
        tf.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    augmentation = tf.Compose([
        preprocess,
        tf.RandomHorizontalFlip(),
        tf.RandomCrop(32, padding=4),
    ])

    train_data = CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=augmentation
    )

    test_data = CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=preprocess
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

if __name__ == '__main__':
    import pickle

    import matplotlib.pyplot as plt
    
    with open('datasets\cifar-100-python\meta', 'rb') as f:
        label_names = pickle.load(f)

    def denormalize(img):
        img = img * torch.tensor((0.2675, 0.2565, 0.2761)).view(3, 1, 1)
        img = img + torch.tensor((0.5071, 0.4867, 0.4408)).view(3, 1, 1)
        return img
    
    train_loader, test_loader = get_cifar100_datasets(64, 'datasets')

    images, labels = next(iter(train_loader))
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle('Train')
    axes = axes.flatten()
    for idx in range(64):
        img = denormalize(images[idx])
        img = torch.permute(img, (1, 2, 0))
        axes[idx].imshow(img)
        label_i = labels[idx].item()
        title = f'{label_i}: {label_names["fine_label_names"][label_i]}'
        axes[idx].set_title(title)
        axes[idx].axis('off')

    plt.show()

    images, labels = next(iter(test_loader))
    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    fig.suptitle('Test')
    axes = axes.flatten()
    for idx in range(64):
        img = denormalize(images[idx])
        img = torch.permute(img, (1, 2, 0))
        axes[idx].imshow(img)
        label_i = labels[idx].item()
        title = f'{label_i}: {label_names["fine_label_names"][label_i]}'
        axes[idx].set_title(title)
        axes[idx].axis('off')

    plt.show()