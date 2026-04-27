from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_task_dataloaders(task_id, batch_size=64, test_only=False, filter=None):
    """Returns DataLoaders for specific MNIST digit pairs."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Task 0: digits (0,1), Task 1: (2,3), etc.
    digits = [task_id * 2, task_id * 2 + 1]
    
    def filter_idx(dataset):
        if filter is None:
            return [i for i, (_, label) in enumerate(dataset) if label in digits]
        else:
            return [i for i, (_, label) in enumerate(dataset) if label in filter]

    if test_only and filter is not None:
        test_loader = DataLoader(Subset(test_dataset, filter_idx(test_dataset)), batch_size=batch_size, shuffle=False)
        return test_loader
    
    train_loader = DataLoader(Subset(train_dataset, filter_idx(train_dataset)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(test_dataset, filter_idx(test_dataset)), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader