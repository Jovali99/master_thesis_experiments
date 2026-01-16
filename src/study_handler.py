from src.models.mlp_model import MLP3, MLP4
from src.cifar_handler import CifarInputHandler
from src.dataset_handler import get_dataloaders, get_weighted_dataloaders
from src.models.resnet18_model import ResNet18
from src.models.wideresnet28_model import WideResNet
from src.utils import sigmoid_weigths, calculate_logits, rescale_logits, calculate_tauc
from src.save_load import saveTrial, buildTrialMetadata, buildStudyMetadata, saveStudy
from src.tabular_handler import TabularInputHandler
from torch import nn, optim
from LeakPro.leakpro.attacks.mia_attacks.rmia import rmia_get_gtlprobs, rmia_vectorised
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import optuna
import os

# Define the datasets
train_dataset: CifarInputHandler.UserDataset | TabularInputHandler.TabularUserDataset | None = None
test_dataset: CifarInputHandler.UserDataset | TabularInputHandler.TabularUserDataset | None = None

def train_one_epoch(model, optimizer, train_loader, device, epoch, epochs):
    model.train()
    for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, position=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return correct / total

def objective(trial, config, device):
    #--------- Hyperparameters ---------
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    T_max = trial.suggest_int("T_max", 25, config["study"]["epochs"], step=5)

    # --------- Dataset setup ---------
    dataset_name = config["data"]["dataset"]
    targets = train_dataset.dataset.targets
    n_classes = int(torch.max(targets).item()) + 1
    input_dim = train_dataset.dataset.data.shape[1]

    train_loader, val_loader = get_dataloaders(batch_size, train_dataset, test_dataset)
    
    # --------- Model setup ---------
    model_name = config["study"]["model"]
    if model_name == "resnet":
        model = torchvision.models.resnet18(num_classes=n_classes).to(device)
        print(f"Optimizing baseline resnet on dataset {dataset_name} n_classes: {n_classes}")

    elif model_name == "wideresnet":
        drop_rate = trial.suggest_float("drop_rate", 0.0, 0.5)
        model = WideResNet(depth=28, num_classes=n_classes, widen_factor=10, dropRate=drop_rate).to(device)
        print(f"Optimizing baseline wideresnet on dataset {dataset_name} with n_classes: {n_classes}")

    elif model_name == "mlp3":
        model = MLP3(input_dim=input_dim, num_classes=n_classes).to(device)
        print(f"Optimizing baseline MLP3 on dataset {dataset_name}, n_classes: {n_classes}, input_dim: {input_dim}")

    elif model_name == "mlp4":
        drop_rate = trial.suggest_float("drop_rate", 0.0, 0.5)
        model = MLP4(input_dim=input_dim, num_classes=n_classes, dropout=drop_rate).to(device)
        print(f"Optimizing baseline MLP4 on dataset {dataset_name}, n_classes: {n_classes}, input_dim: {input_dim}")

    else:
        raise ValueError(f"Invalid model selection{dataset_name}")

    # --------- Optimizer setup ---------
    optimizer_name = config['study']['optimizer']
        
    if config['study']['model'] == "mlp3" or config['study']['model'] == "mlp4":
        optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])

    if optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.8, 0.99)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    # --------- Training loop ---------
    max_epochs = config["study"]["epochs"]
    best_val_accuracy = 0.0
    for epoch in tqdm(range(max_epochs), desc="Training Progress"):
        train_one_epoch(model, optimizer, train_loader, device, epoch, max_epochs)
        scheduler.step()

        val_accuracy = evaluate(model, val_loader, device)

        print(f"Trial val accuracy: {val_accuracy}")
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        if best_val_accuracy < val_accuracy:
            best_val_accuracy = val_accuracy

    return best_val_accuracy

def run_baseline_optimization(config, gpu_id, trials, save_path, hash_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    study_cfg = config['study']

    # Parallell storage setup
    db_path = os.path.join(study_cfg['root'], "baseline_study.db")
    storage = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name=f"{study_cfg['study_name']}-{hash_id}",
        storage=storage,
        load_if_exists=True,
        direction="maximize"
    )
    def safe_obj(trial):
        try:
            return objective(trial, config, device)
        except (optuna.exceptions.StorageInternalError, RuntimeError, OSError, ValueError) as e:
            print(f"[Optuna][GPU {gpu_id}] Trial {trial.number} failed safely: {e}")
            raise optuna.exceptions.TrialPruned()
    
    study.optimize(safe_obj, n_trials=trials)
    
    print(f"Study '{study_cfg['study_name']}' completed on GPU {gpu_id}. Best value: {study.best_value}, params: {study.best_params}")
    df = study.trials_dataframe() 
    df.to_csv(os.path.join(save_path, f"results_gpu_{gpu_id}.csv"), index=False) 
    print(f"📄 Results saved to {os.path.join(save_path, f'results_gpu_{gpu_id}.csv')}")

    return study


def fbd_objective(trial, cfg, rmia_scores, train_dataset, test_dataset, shadow_gtl_probs, shadow_inmask, target_inmask, tauc_ref, save_path, device):
    """
        noise_std: Trial between [0.0001, 0.05] step = 0.005
        Centrality: Trial stepped between [0.0, 1.0] step = 0.1
        Temperature: Trial between [0.000 + 1e-6, 0.5] step = 0.05
    """
    # ------------ study params setup ------------
    if cfg['fbd_study']["privacy_onion"]:
        noise_std = 0.0
        centrality = trial.suggest_float("centrality", 0.0, 1.0, step=0.05)
        temperature = 0.0
    elif cfg['fbd_study']["noise_injection"]:
        levels = [10, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0001]
        noise_std = trial.suggest_categorical("noise_std", levels)
        centrality = 1.0
        temperature = 0.0
    elif cfg['fbd_study']["strict_fbd"]:
        noise_std = trial.suggest_float("noise_std", 0.0, 0.025, step=0.005)
        centrality = trial.suggest_float("centrality", 0.0, 1.0, step=0.1)
        temperature = trial.suggest_float("temperature", 0.0, 0.25, step=0.05)
    else:
        noise_std = trial.suggest_float("noise_std", 0.0, 0.05, step=0.005)
        centrality = trial.suggest_float("centrality", 0.0, 1.0, step=0.1)
        temperature = trial.suggest_float("temperature", 0.0, 0.5, step=0.05)

    # Calculate the weights
    weights = sigmoid_weigths(rmia_scores, centrality, temperature)

    assert train_dataset.dataset is test_dataset.dataset, "train_dataset.dataset =/= test_dataset.dataset"

    lr = cfg["fbd_study"]["learning_rate"]
    weight_decay = cfg["fbd_study"]["weight_decay"]
    epochs = cfg["fbd_study"]["epochs"]
    momentum = cfg["fbd_study"]["momentum"]
    t_max = cfg["fbd_study"]["t_max"]
    batch_size = cfg["fbd_study"]["batch_size"]
    optim_name = cfg["fbd_study"]["optimizer"]
    drop_rate = cfg["fbd_study"]["drop_rate"]

    # --------- Dataset setup ---------
    dataset_name = cfg["data"]["dataset"]
    targets = train_dataset.dataset.targets
    n_classes = int(torch.max(targets).item()) + 1
    input_dim = train_dataset.dataset.data.shape[1]
    
    # --------- Model setup ---------
    model_name = cfg["fbd_study"]["model"]
    if model_name == "resnet":
        model = torchvision.models.resnet18(num_classes=n_classes).to(device)
    elif model_name == "wideresnet":
        model = WideResNet(depth=28, num_classes=n_classes, widen_factor=10, dropRate=drop_rate).to(device)
    elif model_name == "mlp3":
        model = MLP3(input_dim=input_dim, num_classes=n_classes).to(device)
    elif model_name == "mlp4":
        model = MLP4(input_dim=input_dim, num_classes=n_classes, dropout=drop_rate).to(device)
    else:
        raise ValueError(f"Invalid model selection{dataset_name}")
    print(f"Optimizing FbD {model_name} on dataset {dataset_name} n_classes: {n_classes}")

    # ------------ Choose Optimizer ------------ #
    if optim_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss(reduction="none")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    # ------------ Get the Dataloaders ------------ #
    train_loader, test_loader = get_weighted_dataloaders(batch_size, train_dataset, test_dataset, weights)

    # ------------ TRAIN MODEL ------------ #
    if model_name in ["resnet", "wideresnet"]:
        handler = CifarInputHandler()
    else:
        handler = TabularInputHandler()

    handler.trainStudyFbD(train_loader, model, criterion, optimizer, epochs, noise_std, scheduler)
    test_accuracy = handler.eval(test_loader, model, criterion).accuracy

    # ------------ Calculate Logits and GTL Probs ------------ #
    full_dataset = train_dataset.dataset

    model.to(device)

    target_logits = calculate_logits(model, full_dataset, device)
    labels = np.array(full_dataset.targets)
    rescaled_target_logits = rescale_logits(target_logits, labels)
    target_gtl_probs = rmia_get_gtlprobs(target_logits, labels)
    scores = rmia_vectorised(target_gtl_probs, shadow_gtl_probs, shadow_inmask, online=True, use_gpu_if_available=True)

    model.to("cpu")

    tauc_weighted = calculate_tauc(scores, target_inmask, fpr=0.1)
    tau = np.log(tauc_weighted/tauc_ref)

    # ------------ SAVE RESULTS ------------ #
    metadata = buildTrialMetadata(noise_std, centrality, temperature, test_accuracy, tau)
    saveTrial(metadata, target_gtl_probs, rescaled_target_logits, trial.number, save_path)

    return tau, test_accuracy
