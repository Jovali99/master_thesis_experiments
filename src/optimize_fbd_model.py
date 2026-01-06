from dataclasses import dataclass
import optuna
import pandas as pd
import numpy as np
import yaml
import pickle
import torch
import os
import matplotlib.pyplot as plt
import multiprocessing

import src.save_load as sl

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch import tensor, cat, save, load, optim, nn
from torch.utils.data import DataLoader
from src.models.resnet18_model import ResNet18

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import src.study_handler as sh
from src.utils import print_yaml, get_shadow_signals, calculate_tauc
from LeakPro.leakpro.attacks.mia_attacks.rmia import rmia_vectorised, rmia_get_gtlprobs
from src.save_load import loadTargetSignals, loadShadowModelSignals, copy_study_to_global
from src.dataclasses import FbdArgs

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

def run_optimization(config, gpu_id, trials, save_path, hash_id, fbd_args: FbdArgs, node_id: int | None = None): 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    study_cfg = config['fbd_study']

    # Parallell storage setup
    global_db_path = os.path.join(study_cfg['root'], "fbd_study.db")
    if node_id is None:
        storage = f"sqlite:///{global_db_path}"
        print("Running study on Global optuna database")
    else:
        # Use a node specific db
        node_db_path = os.path.join(study_cfg['root'], f"fbd_study_{node_id}.db")
        storage = f"sqlite:///{node_db_path}"
        print(f"Running study on Node-{node_id} optuna database")
    
    study = optuna.create_study(
        study_name=f"{study_cfg['study_name']}-{hash_id}",
        storage=storage,
        load_if_exists=True,
        directions=["minimize", "maximize"]
    )
    # Extract arguments from dataclass
    rmia_scores = fbd_args.rmia_scores
    train_dataset = fbd_args.train_dataset
    test_dataset = fbd_args.test_dataset
    shadow_gtl_probs = fbd_args.shadow_gtl_probs
    shadow_inmask = fbd_args.shadow_inmask
    target_inmask = fbd_args.target_inmask
    tauc_ref = fbd_args.tauc_ref

    def safe_obj(trial):
        try:
            return sh.fbd_objective(trial, config, rmia_scores, train_dataset, 
                                                  test_dataset, shadow_gtl_probs, shadow_inmask, 
                                                  target_inmask, tauc_ref, save_path, device)
        except (optuna.exceptions.StorageInternalError, RuntimeError, OSError, ValueError) as e:
            print(f"[Optuna][GPU {gpu_id}] Trial {trial.number} failed safely: {e}")
            raise optuna.exceptions.TrialPruned()
    
    study.optimize(safe_obj, n_trials=trials)
    
    print(f"Study '{study_cfg['study_name']}' completed on GPU {gpu_id}.")
    df = study.trials_dataframe() 
    df.to_csv(os.path.join(save_path, f"results_gpu_{gpu_id}.csv"), index=False) 
    print(f"📄 Results saved to {os.path.join(save_path, f'results_gpu_{gpu_id}.csv')}")
    
    if node_id is not None:
        copy_study_to_global(
            node_db_path=node_db_path,
            global_db_path=global_db_path,
            study_name=f"{study_cfg['study_name']}-{hash_id}",
        )

def parallell_optimization(config, labels, fbd_args, gpu_ids = [0], study_hash: str | None = None, node_id: int | None = None):
    study_cfg = config['fbd_study']
    print(f"Starting parallell optimization using the following gpu ids: {gpu_ids}")

    if study_hash is not None:
        hash_id = study_hash  # Enter the hash of the 
        study_name = f"{study_cfg['study_name']}-{hash_id}"
        save_path = os.path.join("study", study_name)
    else:
        metadata = sl.buildStudyMetadata(study_cfg, config['data']) 
        hash_id, save_path = sl.saveStudy(metadata, savePath=study_cfg['root'], labels=labels)
    
    # split up the trials among the gpus
    total_trials = study_cfg["trials"]
    n_gpus = len(gpu_ids)
    base_trials, remainder = divmod(total_trials, n_gpus)
    
    # assign trials per GPU
    trials_per_gpu = [base_trials + 1 if i < remainder else base_trials for i in range(n_gpus)]

    processes = []
    for gpu_id, trials in zip(gpu_ids, trials_per_gpu):
        print(f"Running {trials} on gpu: {gpu_id}")
        p = multiprocessing.Process(
            target=run_optimization,
            args=(config, gpu_id, trials, save_path, hash_id, fbd_args, node_id)
        )
        processes.append(p)
    
    for p in processes:
        p.start() 
    for p in processes:
        p.join()
        
    db_path = os.path.join(study_cfg['root'], "fbd_study.db")
    storage = f"sqlite:///{db_path}"
    study_name = f"{study_cfg['study_name']}-{hash_id}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study
