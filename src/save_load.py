import hashlib
import os
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from src.dataclasses import FbdTrialResults
import pickle
import optuna

def buildAuditMetadata(trainCfg: dict, auditCfg: dict = {}) -> dict:
    """
    Construct metadata describing the training configuration, hyperparameters, 
    and (optionally) audit configuration.
    """
    metadata = {
        "trainCfg": trainCfg,
        "auditCfg": auditCfg
    }
    return metadata

def buildTargetMetadata(trainCfg: dict, dataCfg: dict, additionalCfg: dict = {}) -> dict:
    """
    Construct metadata describing the study configuration. 
    """
    metadata = {
        "train": trainCfg,
        "data": dataCfg,
        "additionalCfg": additionalCfg if additionalCfg is not None else {}
    }
    return metadata

def buildStudyMetadata(studyCfg: dict, dataCfg: dict, additionalCfg: dict = {}) -> dict:
    """
    Construct metadata describing the study configuration. 
    """
    metadata = {
        "study": studyCfg,
        "data": dataCfg,
        "additionalCfg": additionalCfg if additionalCfg is not None else {}
    }
    return metadata

def buildTrialMetadata(noise, centrality, temperature, accuracy, tau):
    metadata = {
        "noise": noise,
        "centrality": centrality,
        "temperature": temperature,
        "accuracy": accuracy,
        "tau@0.1": tau,
    }
    return metadata

def hashCfg(metadata:dict, inmask: np.ndarray = None) -> str:
    """
    Compute a unique SHA256 hash based on metadata and inmask.
    """
    hash = hashlib.sha256()

    # Hash metadata
    meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
    hash.update(meta_bytes)

    # Hash inmask
    if inmask is not None:
        hash.update(inmask.tobytes())

    return hash.hexdigest()[:10]

def saveAudit(metadata: dict, target_model_logits: np.ndarray,
              shadow_models_logits: np.ndarray, inmask: np.ndarray,
              target_inmask: np.ndarray, audit_data_indices: np.ndarray, savePath:str = "audit_signals"):
    """
    Save a full audit signals into a folder named by its metadata unique hash.
        metadata: the training and audit configuration
        target_logits: Rescaled target model logits
        shadow_models_logits: Rescaled shadow model logits
        inmask: in_indices_mask for the shadow models
        audit_data_indices: indices used for audit
    """
    os.makedirs(savePath, exist_ok=True)

    hash_id = hashCfg(metadata, inmask)

    date_str = datetime.now().strftime("%Y%m%d")

    folder_name = f"{date_str}_{hash_id}"

    save_dir = os.path.join(savePath, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "rescaled_target_logits.npy"), target_model_logits)
    np.save(os.path.join(save_dir, "rescaled_shadow_model_logits.npy"), shadow_models_logits)
    np.save(os.path.join(save_dir, "shadow_models_in_mask.npy"), inmask)
    np.save(os.path.join(save_dir, "target_in_mask.npy"), inmask)
    np.save(os.path.join(save_dir, "audit_data_indices.npy"), audit_data_indices)

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    print(f"✅ Saved target and shadow models logits, inmask and audit data indices with hash_id: {hash_id}")
    return hash_id, save_dir

def saveStudy(metadata: dict, savePath:str = "study", labels: np.ndarray = None):
    """
    Create a uniquely hashed folder for each Optuna study based on its metadata.
    Save metadata.json in that folder and return (hash_id, save_dir).
    """
    os.makedirs(savePath, exist_ok=True)

    hash_id = hashCfg(metadata)
    studyCfg = metadata['study']
    study_name = studyCfg['study_name']

    # Construct save directory
    save_dir = os.path.join(savePath, f'{study_name}-{hash_id}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)
        
    if labels is not None:
        trial_outputs_dir = os.path.join(save_dir, "trial_outputs")
        os.makedirs(trial_outputs_dir, exist_ok=True)
        # Save the labels
        save_labels = os.path.join(trial_outputs_dir, f"labels.npy")
        np.save(save_labels, labels)

    print(f"✅ Saved study journal and study metadata with hash_id: {hash_id}")
    return hash_id, save_dir

def saveTrial(metadata: dict, gtl: np.ndarray, resc_logits: np.ndarray, idx: int, path: str):
    """Saves computed rescaled logits, gtl_probs for the weighted
       target model along with the resulting outputs and the used parameters

    Args:
        metadata (dict): Metadata containing parameters and evaluation outputs
        gtl (np.ndarray): gtl_probabilities used for rmia
        resc_logits (np.ndarray): rescaled_logits used for lira
        idx (int): trial index
        path (str): study/{study_folder}/...

    Returns:
        Confirmation and results
    """
    # Make sure the dir is created if it doesnt exist
    save_dir = os.path.join(path, "trial_outputs")
    
    # Save the metadata to trial_outputs/metadata/...
    md_dir = os.path.join(save_dir, "metadata")
    os.makedirs(md_dir, exist_ok=True)
    with open(os.path.join(md_dir, f"metadata_{idx}.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    # Save the rescaled logits to trial_outputs/rescaled_logits/...
    resc_logits_dir = os.path.join(save_dir, "rescaled_logits")
    os.makedirs(resc_logits_dir, exist_ok=True)
    np.save(os.path.join(resc_logits_dir, f"rescaled_logits_{idx}.npy"), resc_logits)
    
    # Save the gtl probabilities to trial_outputs/gtl_probabilities/...
    gtl_probs_dir = os.path.join(save_dir, "gtl_probabilities")
    os.makedirs(gtl_probs_dir, exist_ok=True)
    np.save(os.path.join(gtl_probs_dir, f"gtl_probabilities_{idx}.npy"), gtl)
    
    return print(f"✅ Saved trial #:{idx} logits and metadata with accuracy {metadata['accuracy']} and tau@0.1 {metadata['tau@0.1']} ")

def saveTarget(metadata: dict, savePath:str = "target"):
    """
    Create a uniquely hashed folder for each target training based on its metadata.
    Save metadata.json in that folder and return (hash_id, save_dir).
    """
    os.makedirs(savePath, exist_ok=True)

    hash_id = hashCfg(metadata)

    # Construct save directory
    model = metadata["train"]["model"]
    dataset = metadata["data"]["dataset"]
    save_dir = os.path.join(savePath, f'{model}-{dataset}-{hash_id}')
    #save_dir = os.path.join(savePath, f'{"resnet18"}-{hash_id}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    print(f"✅ Saved training metadata with hash_id: {hash_id}")
    return hash_id, save_dir

def saveTargetSignals(target_model_logits: np.ndarray, in_mask: np.ndarray, path:str, resc_logits: np.ndarray = None, gtl_probs: np.ndarray = None):
    """
    Saves the logits and in_mask of a target model and optionally resc_logits and gtl_probs
    """
    # Save logits
    save_logits_path = os.path.join(path, "target_logits.npy")
    np.save(save_logits_path, target_model_logits)
    
    if(resc_logits is not None):
        save_resc_logits_path = os.path.join(path, "target_rescaled_logits.npy")
        np.save(save_resc_logits_path, resc_logits)
    if(gtl_probs is not None):
        save_gtl_probs_path = os.path.join(path, "target_gtl_probs.npy")
        np.save(save_gtl_probs_path, gtl_probs)

    # Save in_mask
    save_in_mask_path = os.path.join(path, f"target_in_mask.npy")
    np.save(save_in_mask_path, in_mask)
    print(f"✅ Saved target model logits at: {save_logits_path}, and in mask at {save_in_mask_path}")

def saveShadowModelSignals(identifier: int, logits: np.ndarray=None, in_mask: np.ndarray=None, 
                           resc_logits: np.ndarray=None, gtl_probs: np.ndarray=None, metadata=None, path: str = "processed_shadow_models"):
    """
    Saves the logits and in_mask of a shadow model
    """
    # Ensure all subdirectories exist
    subdirs = ["logits", "in_masks", "rescaled_logits", "gtl_probabilities", "metadata"]
    for sd in subdirs:
        os.makedirs(os.path.join(path, sd), exist_ok=True)
    
    # Save logits
    if logits is not None:
        s_path = os.path.join(path, os.path.join("logits", f"shadow_logits_{identifier}.npy"))
        np.save(s_path, logits)
    
    # Save in_mask
    if in_mask is not None:
        s_path = os.path.join(path, os.path.join("in_masks", f"in_mask_{identifier}.npy"))
        np.save(s_path, in_mask)
        
    # Save resc_logits
    if resc_logits is not None:
        s_path = os.path.join(path, os.path.join("rescaled_logits", f"resc_logits_{identifier}.npy"))
        np.save(s_path, resc_logits)
        
    # Save gtl_probs
    if gtl_probs is not None:
        s_path = os.path.join(path, os.path.join("gtl_probabilities", f"gtl_probs_{identifier}.npy"))
        np.save(s_path, gtl_probs)
    
    # Save metadata
    if metadata is not None:
        s_path = os.path.join(path, os.path.join("metadata", f"metadata_{identifier}.pkl"))
        with open(s_path, "wb") as f:
            pickle.dump(metadata, f)

    print(f"✅ Saved shadow model signals at: {path}")

def saveVisData(data: np.ndarray, title: str, study_name: str, path: str = "study"):
    save_path = os.path.join(os.path.join(path, study_name), "visualization_data")
    os.makedirs(save_path, exist_ok=True)
    f_path = os.path.join(save_path, title)
    np.save(f_path, data)
    return print(f"Visualization data: {title}, with shape: {data.shape}, has been saved to: {f_path}.")

def loadVisData(study_name, study_path: str = "study"):
    """
    Load all visualization data stored under <study_name>/<path>.

    Returns:
        dict[str, np.ndarray]: {title: data}
    """

    base_path = os.path.join(study_path, os.path.join(study_name, "visualization_data"))
    os.makedirs(base_path, exist_ok=True)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Visualization path does not exist: {base_path}")

    data_dict = {}

    for fname in os.listdir(base_path):
        if fname.endswith(".npy"):
            title = os.path.splitext(fname)[0]
            file_path = os.path.join(base_path, fname)

            data = np.load(file_path, allow_pickle=True)
            data_dict[title] = data

            print(f"Loaded {title} with shape {data.shape}")

    if len(data_dict) == 0:
        print("⚠️ No visualization files found.")

    return data_dict

def loadTargetSignals(target_name: str, path: str = "target"):
    """ Loads the target logits, in_mask and metadata from input path """

    target_dir = os.path.join(path, target_name)
    target_logits_path = os.path.join(target_dir, "target_logits.npy")
    target_inmask_path = os.path.join(target_dir, "target_in_mask.npy")
    target_resc_logits_path = os.path.join(target_dir, "target_rescaled_logits.npy")
    target_gtl_probs_path = os.path.join(target_dir, "target_gtl_probs.npy")
    target_metadata_path = os.path.join(target_dir, "metadata.json")
    model_metadata_pkl_path = os.path.join(target_dir, "model_metadata.pkl")

    target_logits = np.load(target_logits_path)
    target_inmask = np.load(target_inmask_path)
    resc_logits = np.load(target_resc_logits_path)
    gtl_probs = np.load(target_gtl_probs_path)

    # Load metadata
    with open(target_metadata_path, "r") as f:
        metadata = json.load(f)
        
    with open(model_metadata_pkl_path, "rb") as f:
        metadata_pkl = pickle.load(f)

    if target_logits is not None:
        print(f"✅ Target logits loaded, shape: {target_logits.shape}")
    if target_inmask is not None:
        print(f"✅ Target inmask loaded, shape: {target_inmask.shape}")
    if resc_logits is not None:
        print(f"✅ Target resc_logits loaded, shape: {resc_logits.shape}")
    if gtl_probs is not None:
        print(f"✅ Target gtl_probs loaded, shape: {gtl_probs.shape}")

    print(f"loaded from: {target_dir}")
    return target_logits, target_inmask, resc_logits, gtl_probs, metadata, metadata_pkl

def loadShadowModelSignals(target_name: str, load_dict: dict = None, path: str = "processed_shadow_models", expected_sm: int = 256):
    # Default: load everything
    if load_dict is None:
        load_dict = {
            "logits": True,
            "resc_logits": True,
            "gtl_probs": True,
            "in_mask": True,
            "metadata_pkl": True
        }

    base_dir = os.path.join(path, target_name)
    assert os.path.exists(base_dir), f"Base shadow path does not exist: {base_dir}"

    
    # Storage lists for stacking
    logits_list = []
    resc_logits_list = []
    gtl_probs_list = []
    inmask_list = []
    metadata_list = []
    missing_indices = []


    for index in range(expected_sm):
        paths = {
            "logits":        os.path.join(base_dir, "logits",             f"shadow_logits_{index}.npy"),
            "resc_logits":   os.path.join(base_dir, "rescaled_logits",    f"resc_logits_{index}.npy"),
            "gtl_probs":     os.path.join(base_dir, "gtl_probabilities",  f"gtl_probs_{index}.npy"),
            "in_mask":       os.path.join(base_dir, "in_masks",           f"in_mask_{index}.npy"),
            "metadata_pkl":  os.path.join(base_dir, "metadata",           f"metadata_{index}.pkl")
        }

        found = False
        # Load each requested component
        if load_dict.get("logits", False) and os.path.exists(paths["logits"]):
            logits_list.append(np.load(paths["logits"]))
            found = True

        if load_dict.get("resc_logits", False) and os.path.exists(paths["resc_logits"]):
            resc_logits_list.append(np.load(paths["resc_logits"]))
            found = True

        if load_dict.get("gtl_probs", False) and os.path.exists(paths["gtl_probs"]):
            gtl_probs_list.append(np.load(paths["gtl_probs"]))
            found = True

        if load_dict.get("in_mask", False) and os.path.exists(paths["in_mask"]):
            inmask_list.append(np.load(paths["in_mask"]))
            found = True

        if load_dict.get("metadata_pkl", False) and os.path.exists(paths["metadata_pkl"]):
            with open(paths["metadata_pkl"], "rb") as f:
                metadata_list.append(pickle.load(f))
            found = True

        if not found:
            missing_indices.append(index)

    print(f"✅ Loaded up to index {expected_sm}")
    print(f"Missing indices: {missing_indices}")

    sm_logits = (
        np.stack(logits_list, axis=1)
        if load_dict.get("logits", False) and len(logits_list) > 0
        else False
    )

    sm_resc_logits = (
        np.stack(resc_logits_list, axis=1)
        if load_dict.get("resc_logits", False) and len(resc_logits_list) > 0
        else False
    )

    sm_gtl_probs = (
        np.stack(gtl_probs_list, axis=1)
        if load_dict.get("gtl_probs", False) and len(gtl_probs_list) > 0
        else False
    )

    sm_in_masks = (
        np.stack(inmask_list, axis=1)
        if load_dict.get("in_mask", False) and len(inmask_list) > 0
        else False
    )

    sm_metadata = metadata_list if len(metadata_list) > 0 else False

    if sm_logits is not False:
        print(f"➡️ Logits shape: {sm_logits.shape}")
    if sm_in_masks is not False:
        print(f"➡️ In-mask shape: {sm_in_masks.shape}")
    if sm_resc_logits is not False:
        print(f"➡️ resc_logits shape: {sm_resc_logits.shape}")
    if sm_gtl_probs is not False:
        print(f"➡️ gtl_prob shape: {sm_gtl_probs.shape}")

    return sm_logits, sm_resc_logits, sm_gtl_probs, sm_in_masks, sm_metadata, missing_indices

def loadFbdStudy(study_name: str, metadata: bool = True, gtl: bool = True, logits: bool = True, path: str = "study"):
    """
    Load FBD study output files from the study/<study_name>/trial_outputs directory.
    Files must follow the indexed naming scheme:
      metadata/metadata_{i}.json
      gtl_probabilities/gtl_probabilities_{i}.npy
      rescaled_logits/rescaled_logits_{i}.npy
    """
    study_dir = os.path.join(path, study_name)
    trial_outputs_dir = os.path.join(study_dir, "trial_outputs")
    
    # Load labels
    labels_dir = os.path.join(trial_outputs_dir, "labels.npy")
    labels = np.load(labels_dir)

    meta_dir = os.path.join(trial_outputs_dir, "metadata")
    gtl_dir = os.path.join(trial_outputs_dir, "gtl_probabilities")
    logits_dir = os.path.join(trial_outputs_dir, "rescaled_logits")
    
    global_metadata_path = os.path.join(study_dir, "metadata.json")
    if os.path.isfile(global_metadata_path):
        with open(global_metadata_path, "r") as f:
            global_metadata = json.load(f)
    else:
        global_metadata = None

    meta_indices = sorted(int(f.split("_")[1].split(".")[0]) for f in os.listdir(meta_dir) if f.startswith("metadata_") and f.endswith(".json"))

    fbd_trial_results = []
    gtl_list = []
    logits_list = []

    for idx in meta_indices:
        # Load metadata
        if metadata:
            meta_path = os.path.join(meta_dir, f"metadata_{idx}.json")
            if not os.path.isfile(meta_path):
                print(f"[Warning] Missing metadata file: {meta_path}, skipping trial {idx}")
                continue
            with open(meta_path, "r") as f:
                meta_dict = json.load(f)
            trial_result = FbdTrialResults(
                accuracy     = meta_dict["accuracy"],
                noise        = meta_dict["noise"],
                centrality   = meta_dict["centrality"],
                temperature  = meta_dict["temperature"],
                tau          = meta_dict["tau@0.1"]
            )
        else:
            trial_result = None

        # Load GTL
        if gtl:
            gtl_path = os.path.join(gtl_dir, f"gtl_probabilities_{idx}.npy")
            if not os.path.isfile(gtl_path):
                print(f"[Warning] Missing GTL file: {gtl_path}, skipping trial {idx}")
                continue
            gtl_data = np.load(gtl_path)
        else:
            gtl_data = None

        # Load logits
        if logits:
            logits_path = os.path.join(logits_dir, f"rescaled_logits_{idx}.npy")
            if not os.path.isfile(logits_path):
                print(f"[Warning] Missing logits file: {logits_path}, skipping trial {idx}")
                continue
            logits_data = np.load(logits_path)
        else:
            logits_data = None

        # Only append if all requested data is available
        if (metadata and trial_result is None) or (gtl and gtl_data is None) or (logits and logits_data is None):
            continue

        if metadata:
            fbd_trial_results.append(trial_result)
        if gtl:
            gtl_list.append(gtl_data)
        if logits:
            logits_list.append(logits_data)

    print(f"✅ Loaded {len(fbd_trial_results)} trials for study '{study_name}'")
    return global_metadata, fbd_trial_results, gtl_list, logits_list, labels

def loadAudit(audit_signals_name: str, save_path: str = "audit_signals"):
    """
    Load audit data previously saved with saveAudit().
    
    audit_signals_name:
        Folder name of the audit run (e.g. '20250110_123456789')
    save_path:
        Base folder where audit runs are stored.
    
    Returns:
        metadata (dict)
        rescaled_target_logits (np.ndarray)
        rescaled_shadow_model_logits (np.ndarray)
        shadow_models_in_mask (np.ndarray)
        audit_data_indices (np.ndarray)
    """
    audit_dir = os.path.join(save_path, audit_signals_name)

    if not os.path.exists(audit_dir):
        raise FileNotFoundError(f"Audit directory not found: {audit_dir}")

    # --- Load files ---
    metadata_path = os.path.join(audit_dir, "metadata.json")
    target_logits_path = os.path.join(audit_dir, "rescaled_target_logits.npy")
    shadow_logits_path = os.path.join(audit_dir, "rescaled_shadow_model_logits.npy")
    inmask_path = os.path.join(audit_dir, "shadow_models_in_mask.npy")
    target_inmask_path = os.path.join(audit_dir, "target_in_mask.npy")
    indices_path = os.path.join(audit_dir, "audit_data_indices.npy")

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load numpy arrays
    rescaled_target_logits = np.load(target_logits_path)
    rescaled_shadow_model_logits = np.load(shadow_logits_path)
    shadow_models_in_mask = np.load(inmask_path)
    target_in_mask = np.load(target_inmask_path)
    audit_data_indices = np.load(indices_path)

    print(f"📥 Loaded audit signals from folder: {audit_signals_name}")

    return (metadata, rescaled_target_logits, rescaled_shadow_model_logits,
            shadow_models_in_mask, target_in_mask, audit_data_indices)

def savePlot(fig, filename: str, study_name: str, savePath: str = "study", dpi: int = 300, fmt: str = "png"):
    """
    Save a matplotlib figure with high-quality settings.
    
    Parameters:
        fig: matplotlib.figure.Figure
            The figure object to save.
        audit_dir: str
            Name of the audit_signals subdir used for the creation of the plots
        filename: str
            Name of the file without extension.
        savePath: str
            Directory where the figure is stored.
        dpi: int
            Resolution for the exported image.
        fmt: str
            File format ("png", "pdf", "svg", ...).
    """
    plot_path = os.path.join(study_name, "plot")
    save_dir = os.path.join(savePath, plot_path)
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, f"{filename}.{fmt}")

    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    print(f"📁 Saved plot to: {full_path}")

def compute_bootstrap_ci(x, y, bins=50, n_bootstrap=2000, ci=0.95):
    x = np.array(x)
    y = np.array(y)

    # Compute bin edges
    bin_edges = np.linspace(np.min(x), np.max(x), bins + 1)
    bin_idx = np.digitize(x, bin_edges) - 1

    x_means = []
    y_means = []
    y_low_ci = []
    y_high_ci = []

    alpha = (1 - ci) / 2

    for i in range(bins):
        mask = bin_idx == i
        y_bin = y[mask]

        if len(y_bin) < 5:
            continue

        # point estimate
        y_means.append(np.mean(y_bin))
        x_means.append(np.mean(x[mask]))

        # bootstrapping
        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(y_bin, size=len(y_bin), replace=True)
            boot_means.append(sample.mean())
        boot_means = np.array(boot_means)

        y_low_ci.append(np.percentile(boot_means, 100 * alpha))
        y_high_ci.append(np.percentile(boot_means, 100 * (1 - alpha)))

    return (
        np.array(x_means),
        np.array(y_means),
        np.array(y_low_ci),
        np.array(y_high_ci)
    )

def plot_bootstrap_band(ax, x, y, label, color):
    xm, ym, ylow, yhigh = compute_bootstrap_ci(x, y, bins=50)

    ax.plot(xm, ym, color=color, label=label)
    ax.fill_between(xm, ylow, yhigh, color=color, alpha=0.2)
    
def copy_study_to_global(node_db_path, global_db_path, study_name):
    node_storage = f"sqlite:///{node_db_path}"
    global_storage = f"sqlite:///{global_db_path}"

    node_study = optuna.load_study(
        study_name=study_name,
        storage=node_storage,
    )

    global_study = optuna.create_study(
        study_name=study_name,
        storage=global_storage,
        directions=node_study.directions,
        load_if_exists=False,  # guaranteed unique
    )

    for trial in node_study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        global_study.add_trial(
            optuna.trial.create_trial(
                params=trial.params,
                distributions=trial.distributions,
                values=trial.values,
                user_attrs=trial.user_attrs,
                system_attrs=trial.system_attrs,
            )
        )
