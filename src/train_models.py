from src.models.mlp_model import MLP3, MLP4
import numpy as np
from src.tabular_handler import TabularInputHandler
import torch
import torch.nn.functional as F
import torchvision
import os
import pickle
import multiprocessing as mp

from torch import save, load, optim, nn

from src.cifar_handler import CifarInputHandler
from src.cifar_handler import CifarInputHandler
from LeakPro.leakpro import LeakPro
from LeakPro.leakpro.attacks.mia_attacks.rmia import rmia_get_gtlprobs
from src.models.resnet18_model import ResNet18
from src.models.wideresnet28_model import WideResNet
from src.utils import calculate_logits, rescale_logits, get_gtlprobs, calculate_logits_and_inmask
from src.save_load import saveShadowModelSignals, saveTargetSignals
from src.dataset_handler import get_dataloaders, process_dataset_by_indices, build_balanced_dataset_indices

def trainTargetModel(cfg, train_loader, test_loader, train_indices, test_indices, save_dir):
    os.makedirs("target", exist_ok=True)

    # --------- Dataset setup ---------
    dataset_name = cfg["data"]["dataset"]
    targets = train_loader.dataset.dataset.targets
    n_classes = int(torch.max(targets).item()) + 1
    input_dim = train_loader.dataset.dataset.data.shape[1]

    # --------- Model setup ---------
    drop_rate = cfg["train"]["drop_rate"]
    lr = cfg["train"]["learning_rate"]
    weight_decay = cfg["train"]["weight_decay"]
    epochs = cfg["train"]["epochs"]
    momentum = cfg["train"]["momentum"]
    t_max = cfg["train"]["t_max"]

    model_name = cfg["train"]["model"]
    if model_name == "resnet":
        model = torchvision.models.resnet18(num_classes=n_classes)
    elif model_name == "wideresnet":
        model = WideResNet(depth=28, num_classes=n_classes, widen_factor=10, dropRate=drop_rate)
    elif model_name == "mlp3":
        model = MLP3(input_dim=input_dim, num_classes=n_classes)
    elif model_name == "mlp4":
        model = MLP4(input_dim=input_dim, num_classes=n_classes, dropout=drop_rate)
    else:
        raise ValueError(f"Invalid model selection{dataset_name}")

    print(f"====== Training model: {model_name} on dataset: {dataset_name}, n_classes: {n_classes}, input_dim: {input_dim} ======")

    criterion = nn.CrossEntropyLoss()
    if cfg["train"]["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------ Initialize scheduler ------------ #
    if t_max is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    else:
        scheduler = None

    # ------------ TRAIN MODEL ------------ #
    if model_name in ["resnet", "wideresnet"]:
        handler = CifarInputHandler()
    else:
        handler = TabularInputHandler()

    train_result = handler.train(dataloader=train_loader, model=model,
                                             criterion=criterion, optimizer=optimizer,
                                             epochs=epochs, scheduler=scheduler)
    test_result = handler.eval(test_loader, model, criterion)

    model.to("cpu")
    save(model.state_dict(), os.path.join(save_dir, "target_model.pkl"))
    
    # Create and Save LeakPro metadata
    meta_data = LeakPro.make_mia_metadata(train_result = train_result,
                                      optimizer = optimizer,
                                      loss_fn = criterion,
                                      dataloader = train_loader,
                                      test_result = test_result,
                                      epochs = epochs,
                                      train_indices = train_indices,
                                      test_indices = test_indices,
                                      dataset_name = dataset_name)
    metadata_pkl_path = os.path.join(save_dir, "model_metadata.pkl")
    with open(metadata_pkl_path, "wb") as f:
        pickle.dump(meta_data, f)

    # ------------ Calc and Save Logits and GTL Probs ------------ #

    with open(metadata_pkl_path, "rb") as f:
        metadata_pkl = pickle.load(f)
    labels = np.array(targets)
    logits, in_mask = calculate_logits_and_inmask(train_loader.dataset.dataset, model, metadata_pkl, save_dir, idx=None, save=False)
    resc_logits = rescale_logits(logits, labels)
    gtl_probs = rmia_get_gtlprobs(logits, labels)
    saveTargetSignals(logits, in_mask, save_dir, resc_logits, gtl_probs)

    return train_result, test_result

def trainFbDTargetModel(cfg, train_loader, test_loader, train_indices, test_indices, fbd_cfg, mia_type: str):
    print("-- Training model ResNet18 on cifar10  --")
    os.makedirs("target", exist_ok=True)

    if(cfg["data"]["dataset"] == "cifar10"):
        num_classes = 10
    elif(cfg["data"]["dataset"] == "cifar100"):
        num_classes = 100
    else:
        raise ValueError(f"Incorrect dataset {cfg['data']['dataset']}, should be cifar10, cifar 100 or cinic10")

    model = ResNet18(num_classes=num_classes)

    """Parse training configuration"""
    lr = cfg["train"]["learning_rate"]
    weight_decay = cfg["train"]["weight_decay"]
    epochs = cfg["train"]["epochs"]
    momentum = cfg["train"]["momentum"]
    t_max = cfg["train"]["t_max"]
    noise_std = fbd_cfg["noise_std"]

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,)

    # --- Initialize scheduler ---
    if t_max is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    else:
        scheduler = None

    train_result = CifarInputHandler().trainFbD(dataloader=train_loader,
                                             model=model,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             epochs=epochs,
                                             noise_std=noise_std,
                                             scheduler=scheduler)

    test_result = CifarInputHandler().eval(test_loader, model, criterion)

    model.to("cpu")
    model_name = mia_type+"_fbd_target_model.pkl"
    save(model.state_dict(), os.path.join(cfg["run"]["log_dir"], model_name))
    
    # Create and Save LeakPro metadata
    meta_data = LeakPro.make_mia_metadata(train_result = train_result,
                                      optimizer = optimizer,
                                      loss_fn = criterion,
                                      dataloader = train_loader,
                                      test_result = test_result,
                                      epochs = epochs,
                                      train_indices = train_indices,
                                      test_indices = test_indices,
                                      dataset_name = cfg["data"]["dataset"])
    metadata_name = mia_type+"_fbd_model_metadata.pkl"
    metadata_pkl_path = os.path.join(cfg["run"]["log_dir"], metadata_name)
    with open(metadata_pkl_path, "wb") as f:
        pickle.dump(meta_data, f)

    return model, train_result, test_result
    
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

def train_shadow_model(train_cfg, train_dataset, test_dataset, train_indices, test_indices, sm_index, device, full_dataset, target_folder):
    # ------------ SETUP MODEL ------------ #
    epochs = train_cfg["train"]["epochs"]
    lr = train_cfg["train"]["learning_rate"]
    weight_decay = train_cfg["train"]["weight_decay"]
    momentum = train_cfg["train"]["momentum"]
    t_max = train_cfg["train"]["t_max"]
    batch_size = train_cfg["train"]["batch_size"]
    optim_name = train_cfg["train"]["optimizer"]
    drop_rate = train_cfg["train"]["drop_rate"]

    # --------- Dataset setup ---------
    ds_name = train_cfg["data"]["dataset"]
    targets = train_dataset.dataset.targets
    n_classes = int(torch.max(targets).item()) + 1
    input_dim = train_dataset.dataset.data.shape[1]

    m_name = train_cfg["train"]["model"]
    if m_name == "resnet":
        model = torchvision.models.resnet18(num_classes=n_classes).to(device)
    elif m_name == "wideresnet":
        model = WideResNet(depth=28, num_classes=n_classes, widen_factor=10, dropRate=drop_rate).to(device)
    elif m_name == "mlp3":
        model = MLP3(input_dim=input_dim, num_classes=n_classes).to(device)
    elif m_name == "mlp4":
        model = MLP4(input_dim=input_dim, num_classes=n_classes, dropout=drop_rate).to(device)
    print(f"Training {m_name} shadow model {sm_index} on dataset: {ds_name}")
    
    criterion = nn.CrossEntropyLoss()
    if optim_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    
    train_loader, test_loader = get_dataloaders(batch_size, train_dataset, test_dataset)
    
    # ------------ TRAIN MODEL ------------ #
    if m_name == "resnet" or m_name == "wideresnet":
        handler = CifarInputHandler();
    else:
        handler = TabularInputHandler();

    train_result = handler.train(train_loader, model, criterion, optimizer, epochs, scheduler, device)
    sm_model = train_result.model
    test_result = handler.eval(test_loader, model, criterion)
    
    # ------------ SAVE RESULTS ------------ #
    meta_data = LeakPro.make_mia_metadata(train_result = train_result,
                                    optimizer = optimizer,
                                    loss_fn = criterion,
                                    dataloader = train_loader,
                                    test_result = test_result,
                                    epochs = epochs,
                                    train_indices = train_indices,
                                    test_indices = test_indices,
                                    dataset_name = ds_name)
    
    print(f"Calculating logits for shadow model: {sm_index}")
    labels = full_dataset.targets
    logits, in_mask = calculate_logits_and_inmask(full_dataset, sm_model, meta_data, "irrelevant", sm_index, False)
    resc_logits = rescale_logits(logits, labels)
    gtl_probs = get_gtlprobs(logits, labels)
    saveShadowModelSignals(sm_index,
                            #logits=logits,
                            in_mask=in_mask,
                            resc_logits=resc_logits,
                            gtl_probs=gtl_probs,
                            metadata=meta_data,
                            path=os.path.join("processed_shadow_models", target_folder))
        

def create_shadow_models_parallel(train_config, sm_count, gpu_ids, full_dataset, target_folder, train_missing: bool = False, missing_indices: list = []):
    n_gpus = len(gpu_ids)
    
    path = os.path.join("processed_shadow_models", target_folder)
    os.makedirs(path, exist_ok=True)

    index_file = os.path.join(path, "shadow_model_indices.npy")

    # Create a list of balanced dataset_indices per shadow_model
    if not train_missing:
        num_shadow_models = sm_count
        model_indices_per_gpu = [[] for _ in range(n_gpus)]
        for idx in range(num_shadow_models):
            gpu = idx % n_gpus
            model_indices_per_gpu[gpu].append(idx)
        dataset_size = len(full_dataset)
        all_dataset_indices_lists = build_balanced_dataset_indices(num_shadow_models, dataset_size)

        shadow_indices_map = {
            model_id: all_dataset_indices_lists[model_id]
            for model_id in range(num_shadow_models)
        }

        np.save(index_file, shadow_indices_map, allow_pickle=True)
        print(f"Saved shadow model index assignments to: {index_file}")
    else:
        if not os.path.exists(index_file):
            raise FileNotFoundError( f"train_missing=True but no saved index file found at: {index_file}")

        shadow_indices_map = np.load(index_file, allow_pickle=True).item()
        print(f"[INFO] Loaded saved shadow model index assignments from {index_file}")

                # Extract ONLY the missing ones
        missing_indices = sorted(missing_indices)

        model_indices_per_gpu = [[] for _ in range(n_gpus)]

        # Round-robin over missing list itself
        for i, model_id in enumerate(missing_indices):
            gpu = i % n_gpus
            model_indices_per_gpu[gpu].append(model_id)

        # Build dataset lists ONLY for missing models
        all_dataset_indices_lists = {
            mid: shadow_indices_map[mid]
            for mid in missing_indices
        }

    procs = []
    for gpu_id, model_subset in zip(gpu_ids, model_indices_per_gpu):
        print(f"starting sm training on gpu: {gpu_id}, sm indices: {model_subset}")
        dataset_indices_list = [all_dataset_indices_lists[i] for i in model_subset]
        
        p = mp.Process(
            target=batched_shadow_model_creation,
            args=(
                train_config,
                model_subset,
                dataset_indices_list,
                gpu_id,
                full_dataset,
                target_folder
            )
        )
        procs.append(p)
        
    for p in procs:
        p.start() 
    for p in procs:
        p.join()
    return

def batched_shadow_model_creation(train_config, sm_indices, dataset_indices_list, gpu_id, full_dataset, target_folder):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Get the dataset
    
    for sm_index, dataset_indices in zip(sm_indices, dataset_indices_list):
        train_dataset, test_dataset, train_indices, test_indices = process_dataset_by_indices(full_dataset, dataset_indices)

        train_shadow_model(train_config, train_dataset, test_dataset, train_indices, test_indices, sm_index, device, full_dataset, target_folder)
