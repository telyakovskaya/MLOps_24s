import torchvision


ds_train_no_transform = torchvision.datasets.CIFAR10(
    root='./', train=True, transform=None, download=True
)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465), 
        std=(0.2023, 0.1994, 0.2010)
    )
])

dataset_cifar10_train = torchvision.datasets.CIFAR10(
    root='./', train=True, transform=transform, download=True
)
dataset_cifar10_test = torchvision.datasets.CIFAR10(
    root='./', train=False, transform=transform, download=True
)