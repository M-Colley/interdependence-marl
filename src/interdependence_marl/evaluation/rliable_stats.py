"""Rigorous Deep RL Statistical Evaluation (`rliable`-style, Agarwal et al., NeurIPS 2021).

Implements Interquartile Mean (IQM), Stratified Percentile Bootstrap 95% Confidence
Intervals, and Probability of Improvement `P(X > Y)` across `[num_seeds, num_tasks]` matrices.
"""

from __future__ import annotations

import numpy as np


def compute_iqm(scores: np.ndarray) -> float:
    """Compute 25% trimmed Interquartile Mean across flattened `(seeds, tasks)` scores."""
    flat = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    n = flat.size
    if n < 4:
        return float(flat.mean())
    low = int(np.floor(0.25 * n))
    high = int(np.ceil(0.75 * n))
    return float(flat[low:high].mean())


def stratified_bootstrap_iqm(
    score_matrix: np.ndarray,
    num_bootstraps: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Compute IQM, Mean, and Stratified Bootstrap CI over `(num_seeds, num_tasks)` matrix."""
    mat = np.asarray(score_matrix, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat[:, None]
    n_seeds, n_tasks = mat.shape
    rng = np.random.default_rng(seed)

    point_iqm = compute_iqm(mat)
    point_mean = float(mat.mean())

    boot_iqms = np.empty(num_bootstraps, dtype=np.float64)
    for b in range(num_bootstraps):
        # Stratified resampling: sample seeds with replacement independently within each task
        idx = rng.integers(0, n_seeds, size=(n_seeds, n_tasks))
        resampled = np.take_along_axis(mat, idx, axis=0)
        boot_iqms[b] = compute_iqm(resampled)

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(boot_iqms, [alpha, 1.0 - alpha])
    return {
        "iqm": point_iqm,
        "mean": point_mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def probability_of_improvement(scores_x: np.ndarray, scores_y: np.ndarray) -> float:
    """Compute Mann-Whitney U Probability of Improvement `P(X > Y)` averaged across tasks."""
    x = np.asarray(scores_x, dtype=np.float64)
    y = np.asarray(scores_y, dtype=np.float64)
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    n_tasks = x.shape[1]
    probs = []
    for t in range(n_tasks):
        diff = x[:, t][:, None] - y[:, t][None, :]
        p_t = float((diff > 0).sum() + 0.5 * (diff == 0).sum()) / float(diff.size)
        probs.append(p_t)
    return float(np.mean(probs))
