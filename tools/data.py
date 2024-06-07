import subprocess

import torch
import torchvision


def get_dataloaders(root, train_batch_size, test_batch_size, num_workers):
    subprocess.run(["dvc", "pull"])

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    dataset_cifar10_train = torchvision.datasets.CIFAR10(
        root=root, train=True, transform=transform, download=False
    )
    dataset_cifar10_test = torchvision.datasets.CIFAR10(
        root=root, train=False, transform=transform, download=False
    )

    dl_train = torch.utils.data.DataLoader(
        dataset_cifar10_train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dl_test = torch.utils.data.DataLoader(
        dataset_cifar10_test, batch_size=test_batch_size, num_workers=num_workers
    )
    return (dl_train, dl_test)