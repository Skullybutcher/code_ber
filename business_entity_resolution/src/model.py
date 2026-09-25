"""
Pairwise GBDT matcher trained with grouped (by S1 entity) cross-validation,
per-entity-normalized sample weights so the macro-averaged evaluation
metric is what training actually reflects, out-of-fold probability
calibration, and a threshold search that directly optimizes entity-level
macro F0.5 rather than a generic classification metric.
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

from metric import macro_f05
from features import FEATURE_NAMES


def make_sample_weights(s1_ids: pd.Series) -> np.ndarray:
    """weight_i = 1 / (# pairs sharing this pair's S1 entity)."""
    counts = s1_ids.value_counts()
    return (1.0 / s1_ids.map(counts)).to_numpy()


def train_oof(
    feat_df: pd.DataFrame,
    labels: np.ndarray,
    n_splits: int = 5,
    lgb_params: dict | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, List[lgb.Booster]]:
    """
    feat_df must contain FEATURE_NAMES columns plus 's1_id'.
    Returns (oof_raw_probs, list_of_fold_models).
    """
    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbose=-1,
        seed=seed,
    )
    if lgb_params:
        params.update(lgb_params)

    groups = feat_df["s1_id"].to_numpy()
    X = feat_df[FEATURE_NAMES].to_numpy()
    weights = make_sample_weights(feat_df["s1_id"])

    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(feat_df))
    models = []

    for train_idx, val_idx in gkf.split(X, labels, groups):
        train_set = lgb.Dataset(X[train_idx], label=labels[train_idx], weight=weights[train_idx])
        val_set = lgb.Dataset(X[val_idx], label=labels[val_idx], weight=weights[val_idx], reference=train_set)
        booster = lgb.train(
            params, train_set,
            num_boost_round=2000,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[val_idx] = booster.predict(X[val_idx], num_iteration=booster.best_iteration)
        models.append(booster)

    return oof, models


def calibrate_oof(oof_raw: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_raw, labels)
    return iso


def _predictions_at_threshold(
    s1_ids: np.ndarray, other_ids: np.ndarray, probs: np.ndarray, threshold: float
) -> Dict[str, Set[str]]:
    preds: Dict[str, Set[str]] = {}
    mask = probs >= threshold
    for s1, other in zip(s1_ids[mask], other_ids[mask]):
        preds.setdefault(s1, set()).add(other)
    return preds


def tune_threshold(
    feat_df: pd.DataFrame,
    calibrated_probs: np.ndarray,
    gt: Dict[str, Set[str]],
    all_s1_ids: List[str],
    grid: np.ndarray | None = None,
) -> Tuple[float, float]:
    """Search a threshold grid, pick the one maximizing macro F0.5 on OOF."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 37)
    s1_arr = feat_df["s1_id"].to_numpy()
    other_arr = feat_df["other_id"].to_numpy()

    best_t, best_score = 0.5, -1.0
    for t in grid:
        preds = _predictions_at_threshold(s1_arr, other_arr, calibrated_probs, t)
        score = macro_f05(gt, preds, entity_ids=all_s1_ids)
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t), float(best_score)


def predict_with_models(models: List[lgb.Booster], X: np.ndarray) -> np.ndarray:
    preds = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
    return preds
