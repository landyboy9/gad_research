import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np

def to_sklearn_arrays(dataset):
    """Convert a torchvision Dataset of grayscale images into
    flattened numpy arrays for scikit-learn: X is [n_samples, H*W], y is [n_samples,]."""
    images = []
    labels = []
    for img, label in dataset:          # img is a [1, H, W] tensor from ToTensor()
        images.append(img.numpy().reshape(-1))  # flatten to [H*W,]
        labels.append(label)

    X = np.stack(images)                # [n_samples, H*W]
    y = np.array(labels)                # [n_samples,]
    return X, y

def random_subset(dataset, n, seed=None):
    """Return a Subset of `dataset` containing `n` randomly chosen samples (no replacement)."""
    if n > len(dataset):
        raise ValueError(f"Requested subset size {n} exceeds dataset size {len(dataset)}")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))

    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, indices)


def get_dataset(name: str, data_dir: str, train: bool = True):
    """Return a torchvision Dataset for the requested dataset name."""
    name = name.lower()
 
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST(
            root=data_dir, train=train, download=True, transform=transform
        )
 
    elif name == "cifar10":
        transform = transforms.Compose([
            # Standard luma conversion: L = 0.2989*R + 0.5870*G + 0.1140*B
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        dataset = datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )
 
    elif name == "svhn":
        # SVHN uses split="train"/"test" instead of a train boolean
        split = "train" if train else "test"
        transform = transforms.Compose([
            # Standard luma conversion: L = 0.2989*R + 0.5870*G + 0.1140*B
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        dataset = datasets.SVHN(
            root=data_dir, split=split, download=True, transform=transform
        )
 
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from mnist, cifar10, svhn.")
 
    return dataset


def load_mnist_subset(dataset_name, n, seed, root):
    X_train, y_train = to_sklearn_arrays(random_subset(get_dataset(dataset_name, root, train=True), n, seed))
    X_test, y_test = to_sklearn_arrays(get_dataset(dataset_name, root, train=False))
    return X_train, y_train, X_test, y_test