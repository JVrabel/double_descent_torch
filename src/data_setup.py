import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(batch_size: int = 64, num_workers: int = 4):
    """Creates training and testing DataLoaders for CIFAR-10.

    Args:
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: Number of subprocesses to use for data loading.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the training data
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes

    return train_dataloader, test_dataloader, class_names