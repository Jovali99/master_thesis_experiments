from LeakPro.leakpro.attacks.mia_attacks.lira import lira_vectorized
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from src.save_load import saveShadowModelSignals, saveTargetSignals
from torch import cat, exp, from_numpy, max, sigmoid, sum

import os
import numpy as np
import torch


def print_yaml(data, indent=0):
    """Recursively print YAML."""
    spacing = "    " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spacing}{key}")
            print_yaml(value, indent + 1)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            print(f"{spacing}- Item {index + 1}:")
            print_yaml(item, indent + 1)
    else:
        print(f"{spacing}{data}")

def percentile_score_normalization(scores: np.ndarray, percentile: int, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize scores using percentile clipping.
    
    Args:
        scores: Array of base model audit scores.
        percentile: Percentile to clip extremes (e.g., 2 will clip lower 2% and upper 98%).
        eps: Small number to prevent division by zero.
        
    Returns:
        Normalized scores in [0, 1].
    """
    if(percentile > 50):
        percentile = 100 - percentile
    lo = np.percentile(scores, percentile)
    hi = np.percentile(scores, 100-percentile)
    norm = (scores - lo) / (hi - lo + eps)
    return np.clip(norm, 0.0, 1.0)

def print_percentiles(threshold, scores):
    """
    prints the percentiles from a set threshold

    args:
        threshold: Value threshold to print percentiles outside it
        scores: The scores, either normalized or normal
    """
    threshold = 5  # change this to whatever you want
    num_above = np.sum(scores > threshold)
    print(f"Number of scores above {threshold}: {num_above}")

    percent_above = 100 * num_above / len(scores)
    print(f"Percentage of scores above {threshold}: {percent_above:.2f}%")

    threshold_2 = -threshold  # change this to whatever you want
    num_below = np.sum(scores < threshold_2)
    print(f"Number of scores below {threshold_2}: {num_below}")

    percent_below = 100 * num_below / len(scores)
    print(f"Percentage of scores above {threshold_2}: {percent_below:.2f}%")

def get_shadow_signals(shadow_logits, shadow_inmask, amount):
    total_models = shadow_logits.shape[1]

    if amount > total_models:
        raise ValueError(f"Requested {amount} shadow models but only {total_models} available.")

    # Choose random shadow model indices
    selected_indices = np.random.choice(total_models, size=amount, replace=False)

    # Select those columns
    logits_sub = shadow_logits[:, selected_indices]
    inmask_sub = shadow_inmask[:, selected_indices]
    return logits_sub, inmask_sub

def sigmoid_weigths(score: np.ndarray, centrality: float, temperature: float, epsilon: float = 1e-6) -> np.ndarray:
    temp = temperature+epsilon
    x = (score-centrality)/temp
    x = np.clip(x, -20, 20)  # Prevent overflow warnings
    exp = np.exp(x)
    weight = 1.0/(1.0+exp)
    return weight

def calculate_logits(model, dataset, device, batch_size=128) -> np.ndarray:
    model.eval()
    logits_list = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for x, _ in tqdm(loader):
            x = x.to(device)
            out = model(x)      # logits
            logits_list.append(out.cpu().numpy())

    logits = np.concatenate(logits_list, axis=0)
    return logits

def rescale_logits(logits, true_label):
    if logits.shape[1] == 1:
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        positive_class_prob = sigmoid(logits).reshape(-1, 1)
        predictions = np.concatenate([1 - positive_class_prob, positive_class_prob], axis=1)
    else:
        predictions = logits - np.max(logits, axis=1, keepdims=True)
        predictions = np.exp(predictions)
        predictions = predictions / np.sum(predictions, axis=1, keepdims=True)

    count = predictions.shape[0]
    y_true = predictions[np.arange(count), true_label]
    predictions[np.arange(count), true_label] = 0
    y_wrong = np.sum(predictions, axis=1)

    output_signals = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    return output_signals

def get_gtlprobs(logits, labels, temperature=1.0, select = None):
    select = np.arange(len(labels)) if select is None else select

    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    assert len(select) == len(labels)
    assert logits.shape[0] > np.max(select)
    assert logits.shape[1] > np.max(labels)
    return softmax_logits(logits, temperature)[select,labels]

def calculate_logits_and_inmask(dataset, model, metadata, path, idx: int | None = None, save: bool = True):
    """
    Calculates logits and the inmask for a given model. If an index is input
    the function assumes the model is a shadow model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if(idx is not None):
        print(f"\n=== Calculating logits and in_mask for shadow model {idx} ===")
    else:
        print(f"\n=== Calculating logits and in_mask for target model ===")

    model = model.to(device)

    logits = calculate_logits(model, dataset, device)
    
    # Create the in_mask from training indices
    in_mask = np.zeros(len(dataset), dtype=np.bool_)

    in_indices = np.array(metadata.train_indices, dtype=np.int64)
    in_mask[in_indices] = True
    if save:
        if(idx is not None):
            saveShadowModelSignals(idx, logits=logits, in_mask=in_mask, path=path)
        else:
            print(f"Logits shape: {logits.shape}")
            print(f"In_mask shape: {in_mask.shape}")
            saveTargetSignals(logits, in_mask, path)
        del logits
        del in_mask
        
    return logits, in_mask

def calculate_roc(scores: np.ndarray, target_inmask: np.ndarray, clip: bool = False, eps: float = 1e-6):
    """
    Compute the ROC curve (TPR–FPR pairs) for membership inference scores.

    Parameters
    ----------
    scores : np.ndarray of shape (N,)
        Attack scores for each data point. Higher values must indicate
        a higher belief that the point is a member of the training set.
    
    target_inmask : np.ndarray of shape (N,)
        Binary membership ground-truth for the target model.
        - 1 = member (sample was included in training)
        - 0 = non-member

    Returns
    -------
    tpr_curve : np.ndarray of shape (N,)
        True Positive Rate values computed at each possible threshold.
    
    fpr_curve : np.ndarray of shape (N,)
        False Positive Rate values computed at each possible threshold.
    """
    idx = np.argsort(scores)[::-1]
    inmask_sorted = target_inmask[idx]

    tp_cum = np.cumsum(inmask_sorted == 1)
    fp_cum = np.cumsum(inmask_sorted == 0)

    positives = np.sum(target_inmask == 1)
    negatives = np.sum(target_inmask == 0)

    tpr_curve = tp_cum / positives 
    fpr_curve = fp_cum / negatives
    if clip:
        fpr_curve = np.clip(fpr_curve, eps, 1.0)
        tpr_curve = np.clip(tpr_curve, eps, 1.0)
        
    return tpr_curve, fpr_curve

def calculate_group_roc(
    scores_list: list[np.ndarray],
    inmask_list: list[np.ndarray],
    fpr_grid: np.ndarray = None,
    clip: bool = True,
    eps: float = 1e-6,
):
    """
    Compute mean ROC curve over multiple models.

    Each element in scores_list and inmask_list corresponds to one model.
    """

    assert len(scores_list) == len(inmask_list)

    if fpr_grid is None:
        fpr_grid = np.logspace(-5, 0, 300)

    tpr_interp_all = []

    for scores, inmask in zip(scores_list, inmask_list):
        tpr, fpr = calculate_roc(scores, inmask, clip=clip, eps=eps)

        # Ensure monotonicity for interpolation
        fpr_unique, idx = np.unique(fpr, return_index=True)
        tpr_unique = tpr[idx]

        tpr_interp = np.interp(fpr_grid, fpr_unique, tpr_unique)
        tpr_interp_all.append(tpr_interp)

    tpr_mean = np.mean(tpr_interp_all, axis=0)
    tpr_std  = np.std(tpr_interp_all, axis=0)

    return fpr_grid, tpr_mean, tpr_std

def calculate_tpr_at_fpr(tpr_curve, fpr_curve, fpr: float = 1.0):
    """
    Compute TPR at a specific FPR via linear interpolation on the ROC curve.

    Parameters
    ----------
    tpr_curve : np.ndarray
        TPR values from `calculate_roc`.
    
    fpr_curve : np.ndarray
        FPR values from `calculate_roc`.
    
    fpr : float, optional (default = 1.0)
        Target false positive rate ∈ [0, 1].

    Returns
    -------
    tpr_at_fpr : float
        The interpolated TPR corresponding to the given FPR target.
    """
    if(fpr > 1.0): fpr = 1.0

    tpr_at_fpr = np.interp(fpr, fpr_curve, tpr_curve)
    return tpr_at_fpr

def calculate_tauc(scores: np.ndarray, target_inmask: np.ndarray, fpr: float = 1.0):
    """
    Compute the Tail Area Under Curve (TAUC) for an ROC Curve up to
    a given FPR threshold.

    Parameters
    ----------
    scores : np.ndarray of shape (N,)
        Attack scores for each data point. Higher values must indicate
        higher confidence that the sample is a member.
    
    target_inmask : np.ndarray of shape (N,)
        Membership mask for the target model.
        - 1 = sample was in the training set
        - 0 = sample was not in the training set
    
    fpr : float, optional (default = 1.0)
        Upper bound on the False Positive Rate ∈ [0, 1] over which the
        partial area is computed. For example:
            fpr=0.1 → AUC over FPR ∈ [0, 0.1].

    Returns
    -------
    TAUC : float
        The partial TAUC computed as:
        
        τ = ∫₀^{fpr} TPR(FPR) d(FPR)
        
        using the trapezoidal rule.
    """
    tpr_curve, fpr_curve = calculate_roc(scores, target_inmask)
    
    mask = fpr_curve <= fpr
    tauc = np.trapz(tpr_curve[mask], fpr_curve[mask])
    return tauc

def calculate_tauc_2(fpr, tpr, f0=0.1):
    # Keep only the region up to f0
    mask = fpr <= f0
    fpr_ = fpr[mask]
    tpr_ = tpr[mask]

    # Ensure we include point exactly at f0
    tpr_f0 = np.interp(f0, fpr, tpr)
    fpr_ = np.append(fpr_, f0)
    tpr_ = np.append(tpr_, tpr_f0)

    # Integrate using trapezoidal rule
    return np.trapz(tpr_, fpr_)

def calculate_tau(scores, target_inmask, fpr=0.1):
    """
    Calculate τ = log(TPR@FPR/FPR)
    """
    tpr_curve, fpr_curve = calculate_roc(scores, target_inmask)
    tpr = calculate_tpr_at_fpr(tpr_curve, fpr_curve, fpr)

    # Regular tau
    tau = np.log(tpr / fpr)
    return tau

def pick_weighted_models(accuracy, taus_rmia, taus_lira, thresholds, margin=2.0):
    accuracy = np.array(accuracy)
    taus_rmia = np.array(taus_rmia)
    taus_lira = np.array(taus_lira)

    all_threshold_indices = []       # all indices matching acc range
    selected_best_indices = []       # final chosen index for each threshold

    for thr in thresholds:

        # Acc in [thr, thr + margin]
        mask = (accuracy >= thr) & (accuracy <= thr + margin)
        idxs = np.where(mask)[0].tolist()

        all_threshold_indices.append(idxs)

        if len(idxs) == 0:
            selected_best_indices.append(None)
            continue

        # --- NEW PART: ensure RMIA & LIRA both have tau values for the same idxs ---
        candidate_idxs = []

        for i in idxs:
            # must have valid RMIA and LIRA taus
            if not np.isnan(taus_rmia[i]) and not np.isnan(taus_lira[i]):
                candidate_idxs.append(i)

        if len(candidate_idxs) == 0:
            selected_best_indices.append(None)
            continue

        # Combine vulnerabilities (lower = more robust)
        combined_tau = taus_rmia[candidate_idxs] + taus_lira[candidate_idxs]

        # Select lowest combined vulnerability
        best_idx = candidate_idxs[np.argmin(combined_tau)]
        selected_best_indices.append(best_idx)

    return all_threshold_indices, selected_best_indices

def compute_smooth_curve(x, y, bins=50):
    x = np.array(x)
    y = np.array(y)

    # Compute bin edges
    bin_edges = np.linspace(np.min(x), np.max(x), bins + 1)

    # Digitize
    bin_idx = np.digitize(x, bin_edges) - 1

    # Compute statistics
    x_means = []
    y_means = []
    y_stds = []

    for i in range(bins):
        mask = bin_idx == i
        if np.sum(mask) < 5:    # skip very small bins
            continue
        x_means.append(x[mask].mean())
        y_means.append(y[mask].mean())
        y_stds.append(y[mask].std())

    return np.array(x_means), np.array(y_means), np.array(y_stds)

def plot_with_band(ax, x, y, label, color):
    xm, ym, ys = compute_smooth_curve(x, y, bins=50)
    
    ax.plot(xm, ym, color=color, label=label)
    ax.fill_between(xm, ym - ys, ym + ys, color=color, alpha=0.2)
    
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

def softmax_logits(logits: np.ndarray, temp:float=1.0, dimension:int=-1) -> np.ndarray:
    """Rescale logits to (0, 1).

    Args:
    ----
        logits ( len(dataset) x ... x nb_classes ): Logits to be rescaled.
        temp (float): Temperature for softmax.
        dimension (int): Dimension to apply softmax.

    """
    # If the number of classes is 1, apply sigmoid to return a matrix of [1 - p, p]
    if logits.shape[dimension] == 1:
        logits = from_numpy(logits)
        positive_confidence = sigmoid(logits / temp)  # Apply sigmoid to get the probability of class 1
        zero_confidence = 1 - positive_confidence     # Probability of class 0
        confidences = cat([zero_confidence, positive_confidence], dim=dimension)  # Stack both confidences
        return confidences.numpy()

    logits = from_numpy(logits) / temp
    logits = logits - max(logits, dim=dimension, keepdim=True).values
    logits = exp(logits)
    logits = logits/sum(logits, dim=dimension, keepdim=True)
    return logits.numpy()
