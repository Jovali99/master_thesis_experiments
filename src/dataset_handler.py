import os
from tabular_handler import TabularInputHandler
import torch
import pickle
import numpy as np

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import datasets
from torchvision import transforms
from torch import tensor, cat, save, load, optim, nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split, ConcatDataset
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

from src.dataclasses import CIFARDatasetStructure
from src.cifar_handler import CifarInputHandler

# Basic dataset class to handle weighted datsets
class weightedDataset(Dataset):
    def __init__(self, dataset, indices, weights):
        self.dataset: Dataset = dataset
        self.weights = weights
        self.indices = indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return self.indices[idx], self.weights[idx], data, label

def loadDataset(data_cfg):
    dataset_name = data_cfg["dataset"]
    root = data_cfg.get("root", data_cfg.get("data_dir"))
    dataset_path = os.path.join(root, dataset_name + ".pkl")

    # If saved dataset exists -> skip raw loading
    if os.path.exists(dataset_path):
        print(f"📦 Found saved dataset: {dataset_path}")
        print("⏩ Skipping raw dataset load (fast restore)")

        with open(dataset_path, "rb") as f:
            full_dataset = pickle.load(f)

        return None, None, full_dataset

    # Otherwise build fresh
    print("📂 No saved dataset found — loading from source.")
    
    transform = transforms.Compose([
       transforms.ToTensor(),  # Convert PIL image to Tensor
    ])

    trainset, testset = None, None
    if(dataset_name == "cifar10"):
        print("⏩ Loading CIFAR-10")
        trainset = CIFAR10(root=root, train=True, download=True, transform=transform)
        testset = CIFAR10(root=root, train=False, download=True, transform=transform)
    elif(dataset_name == "cinic10"):
        print("⏩ Loading CINIC-10")
        trainset, testset = load_cinic()
    elif(dataset_name == "cifar100"):
        print("⏩ Loading CIFAR-100")
        trainset = CIFAR100(root=root, train=True, download=True, transform=transform)
        testset = CIFAR100(root=root, train=False, download=True, transform=transform)
    elif(dataset_name == "purchase100"):
        print("⏩ Loading purchase100")
        full_dataset = load_purchase()
        return None, None, full_dataset
    elif(dataset_name == "texas100"):
        print("⏩ Loading texas100")
        full_dataset = load_texas()
        return None, None, full_dataset
    elif(dataset_name == "location"):
        print("⏩ Loading location")
        full_dataset = load_location()
        return None, None, full_dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    assert trainset != None, "Failed loading the train set"
    assert testset != None, "Failed loading the test set"
    print("-- Dataset loaded: ", dataset_name, " --")
    return trainset, testset, None

def toTensor(trainset, testset):
    train_data = tensor(trainset.data).permute(0, 3, 1, 2).float() / 255
    test_data = tensor(testset.data).permute(0, 3, 1, 2).float() / 255

    train_targets = tensor(trainset.targets)
    test_targets = tensor(testset.targets)

    return train_data, test_data, train_targets, test_targets

def saveDataset(dataset, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(dataset, file)
        print(f"Dataset saved to {file_path}")

def splitDataset(dataset, train_frac, test_frac):
    dataset_size = len(dataset)
    total = train_frac + test_frac
    assert np.isclose(total, 1.0), "Train + Test fractions must sum to 1.0"

    test_size = int(test_frac * dataset_size)

    indices = np.arange(dataset_size)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, shuffle=True)
    return train_idx, test_idx

def processDataset(data_cfg, trainset, testset, in_indices_mask=None, dataset=None):
    f_train = float(data_cfg["f_train"])
    f_test = float(data_cfg["f_test"]) 

    # Tabular datasets will be dicts and need to be converted to dataset objects
    if isinstance(dataset, dict):
        features = dataset['features']
        labels = dataset['labels']
        dataset = TabularInputHandler.TabularUserDataset(features, labels)

    if dataset is None:
        print("-- Processing dataset for training --")

        train_data, test_data, train_targets, test_targets = toTensor(trainset, testset)

        data = cat([train_data.clone().detach(), test_data.clone().detach()], dim=0)
        targets = cat([train_targets, test_targets], dim=0)
        if(data_cfg["dataset"] == "cifar10" or data_cfg["dataset"] == "cifar100"):
            assert len(data) == 60000, " CIFAR-10/100 Population dataset should contain 60000 samples"
        elif(data_cfg["dataset"] == "cinic10"):
            assert len(data) == 270000, "CINIC-10 Population dataset should contain 270000 samples"

        dataset = CifarInputHandler.UserDataset(data, targets)

    # ---------------------------------------------------------------------
    # CASE 1 — Custom train indices given 
    # ---------------------------------------------------------------------
    if in_indices_mask is not None:
        print("Using provided in_indices_mask for training.")

        # Expected sizes (rounded)
        expected_train = int(f_train * len(dataset))
        expected_test = int(f_test * len(dataset))

        # Convert boolean mask → integer array of indices
        assert len(in_indices_mask) == len(dataset), \
            f"in_indices_mask has wrong length: {len(in_indices_mask)} but dataset has {len(dataset)}"

        # Ensure mask is boolean
        assert in_indices_mask.dtype == bool or set(np.unique(in_indices_mask)).issubset({0, 1}), "in_indices_mask must be boolean or contain only 0/1"

        # Extract the actual index positions
        train_indices = np.where(in_indices_mask == 1)[0]

        # Compute test indices = all remaining indices
        all_indices = np.arange(len(dataset))
        test_indices = np.setdiff1d(all_indices, train_indices, assume_unique=False)

        assert len(train_indices) == expected_train, f"Train size mismatch: mask gives {len(train_indices)} but expected {expected_train}"
        assert len(test_indices) == expected_test, f"Test size mismatch: mask gives {len(test_indices)} but expected {expected_test}"
    # ---------------------------------------------------------------------
    # CASE 2 — No custom indices
    # ---------------------------------------------------------------------
    else:
        train_indices, test_indices = splitDataset(dataset, f_train, f_test)

    # Save dataset
    dataset_name = data_cfg["dataset"]
    dataset_root = data_cfg.get("root", data_cfg.get("data_dir"))
    file_path_pkl = os.path.join(dataset_root, dataset_name + ".pkl")
    file_path_npz = os.path.join(dataset_root, dataset_name + ".npz")
    if not os.path.isfile(file_path_pkl) and not os.path.isfile(file_path_npz):
        saveDataset(dataset, file_path_pkl)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # --- Assertion checks ---
    sample_x, sample_y = train_dataset[0]
    assert sample_x.shape == (3, 32, 32), f"Unexpected sample shape: {sample_x.shape}"
    assert not torch.isnan(sample_x).any(), "NaNs found in normalized data"
    assert not torch.isinf(sample_x).any(), "Infs found in normalized data"
    if(data_cfg["dataset"] == "cifar10" or data_cfg["dataset"] == "cinic10"):
        assert 0.0 <= sample_y < 10, f"Target out of range: {sample_y}"
    elif(data_cfg["dataset"] == "cifar100"):
        assert 0.0 <= sample_y < 100, f"Target out of range: {sample_y}"

    print(f"✅ Dataset ready | Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_dataset, test_dataset, train_indices, test_indices

# Used for optuna studies
def get_dataloaders(batch_size, train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def get_weighted_dataloaders(batch_size, train_dataset, test_dataset, weights):
    train_indices = train_dataset.indices
    test_indices = test_dataset.indices

    train_weights = weights[train_indices]
    test_weights = weights[test_indices]

    weighted_train = weightedDataset(train_dataset, train_indices, train_weights)
    weighted_test = weightedDataset(test_dataset, test_indices, test_weights)

    train_loader = DataLoader(weighted_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(weighted_test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def build_balanced_dataset_indices(num_models, dataset_size, seed=123):
    assert num_models % 2 == 0, "Number of shadow models must be even"
    if seed is not None:
        np.random.seed(seed)
    
    A = np.zeros((num_models, dataset_size), dtype=np.uint8)
    all_indices = np.arange(dataset_size)

    for i in range(0, num_models, 2):
        permuted = np.random.permutation(all_indices)
        half = dataset_size // 2
        A[i, permuted[:half]] = 1
        A[i+1, permuted[half:]] = 1

    dataset_indices_lists = [np.where(A[i] == 1)[0] for i in range(num_models)]

    # ---------- Validation asserts ----------
    # Each data point appears in exactly half of the shadow models
    col_sums = np.sum(A, axis=0)
    assert np.all(col_sums == num_models // 2), f"Each data point should appear in exactly half of the models, got {col_sums}"

    # Each shadow model should have roughly k points (allow +/-1 due to rounding)
    row_sums = np.sum(A, axis=1)
    assert np.all(row_sums == half), f"Each shadow model should have exactly {half} points, got {row_sums}"

    return dataset_indices_lists

def process_dataset_by_indices(full_dataset, train_indices):
    all_indices = np.arange(len(full_dataset))
    train_set = set(train_indices)
    test_indices = [i for i in all_indices if i not in train_set]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    return train_dataset, test_dataset, train_indices, test_indices

def imagefolder_to_arrays(img_folder):
    """Converts an ImageFolder dataset to numpy arrays (N, 32, 32, 3) and targets."""
    data = []
    targets = []

    for img, label in tqdm(img_folder, desc="Converting images to numpy"):
        arr = np.array(img.convert("RGB"))
        data.append(arr)
        targets.append(label)

    data = np.stack(data, axis=0)  # (N,32,32,3)
    return data, targets

def load_cinic(root="data/cinic10"):
    """
    Loads CINIC-10 into CIFAR-style data structures.

    train = train + half of valid
    test  = test  + half of valid
    """

    train_path = os.path.join(root, "train")
    test_path  = os.path.join(root, "test")
    valid_path = os.path.join(root, "valid")

    # Load raw image folders (NO transforms)
    train_folder = datasets.ImageFolder(train_path)
    test_folder  = datasets.ImageFolder(test_path)
    valid_folder = datasets.ImageFolder(valid_path)

    # Split valid folder 50/50
    val_len = len(valid_folder)
    half = val_len // 2

    valid_first_half, valid_second_half = random_split(
        valid_folder, [half, val_len - half]
    )

    assert valid_first_half.dataset is valid_folder
    assert valid_second_half.dataset is valid_folder

    # Convert to numpy + targets
    train_data, train_targets = imagefolder_to_arrays(train_folder)
    test_data,  test_targets  = imagefolder_to_arrays(test_folder)

    # Convert halves
    val1_data, val1_targets = imagefolder_to_arrays(valid_first_half)
    val2_data, val2_targets = imagefolder_to_arrays(valid_second_half)

    # Merge into CIFAR-style datasets
    train_data   = np.concatenate([train_data, val1_data], axis=0)
    train_targets = train_targets + val1_targets

    test_data   = np.concatenate([test_data, val2_data], axis=0)
    test_targets = test_targets + val2_targets

    trainset = CIFARDatasetStructure(train_data, train_targets)
    testset  = CIFARDatasetStructure(test_data, test_targets)

    return trainset, testset

def load_purchase():
    dataset = np.load(os.path.join("data", "purchase100.npz"))
    print(f"Shape of features: {dataset['features']}")
    print(f"Shape of labels: {dataset['labels']}")
    return dataset

def load_texas():
    print("❌ Loading of dataset: Texas100 is not implemented")
    dataset = np.load(os.path.join("data", "texas100.npz"))
    return dataset

def load_location():
    print("❌ Loading of dataset: Location is not implemented")
    dataset = np.load(os.path.join("data", "texas100.npz"))
    return dataset
