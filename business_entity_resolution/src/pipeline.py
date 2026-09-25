"""
End-to-end pipeline: load -> validate -> normalize -> block -> featurize ->
train (grouped OOF) -> calibrate -> tune threshold -> run test inference ->
write candidate_pairs.tsv + matching_results.tsv.

Usage (from code/business_entity_resolution/src/):
    python pipeline.py --data-dir /kaggle/input/<your-dataset> \
                        --out-dir /kaggle/working/output
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from io_utils import load_source, load_ground_truth, validate_ground_truth_refs, write_result_tsv
from normalize import normalize_name, normalize_address
from blocking import (generate_candidates, generate_candidates_token,
                      generate_candidates_prefix, union_candidates, candidate_recall)
from features import build_feature_frame_vectorized, FEATURE_NAMES
from model import train_oof, calibrate_oof, tune_threshold, predict_with_models


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_norm_name"] = df["business_name"].map(normalize_name)
    df["_norm_addr"] = df["business_address"].map(normalize_address)
    return df


def build_pair_frame(s1_df, other_df, candidates: Dict[str, Set[str]]) -> pd.DataFrame:
    """Vectorized pair assembly via merges (no per-row .loc)."""
    s1_ids, other_ids = [], []
    for s1_id, others in candidates.items():
        for other_id in others:
            s1_ids.append(s1_id)
            other_ids.append(other_id)
    if not s1_ids:
        return pd.DataFrame(columns=["s1_id", "other_id", "name1", "addr1", "country1",
                                      "name2", "addr2", "country2"])
    pairs = pd.DataFrame({"s1_id": s1_ids, "other_id": other_ids})
    s1_cols = s1_df[["entity_id", "_norm_name", "_norm_addr", "country"]].rename(
        columns={"entity_id": "s1_id", "_norm_name": "name1", "_norm_addr": "addr1", "country": "country1"})
    other_cols = other_df[["entity_id", "_norm_name", "_norm_addr", "country"]].rename(
        columns={"entity_id": "other_id", "_norm_name": "name2", "_norm_addr": "addr2", "country": "country2"})
    return pairs.merge(s1_cols, on="s1_id", how="left").merge(other_cols, on="other_id", how="left")


def subsample_negatives(feat_df: pd.DataFrame, labels: np.ndarray, neg_per_pos_cap: int,
                        seed: int = 42) -> tuple:
    """Keep all positives; cap negatives per S1 entity. Training-only, not recall."""
    if len(feat_df) == 0:
        return feat_df, labels
    rng = np.random.default_rng(seed)
    df = feat_df.copy()
    df["_label"] = labels
    keep = []
    for _, grp in df.groupby("s1_id"):
        pos = grp.index[grp["_label"] == 1].to_numpy()
        neg = grp.index[grp["_label"] == 0].to_numpy()
        keep.extend(pos.tolist())
        cap = max(neg_per_pos_cap * max(len(pos), 1), 5)
        if len(neg) > cap:
            neg = rng.choice(neg, size=cap, replace=False)
        keep.extend(neg.tolist())
    keep = np.array(sorted(keep), dtype=np.int64)
    return feat_df.loc[keep].reset_index(drop=True), labels[keep]


def run(data_dir: str, out_dir: str, n_splits: int = 5, seed: int = 42,
        sample_s1: int | None = None, max_df: int = 300, prefix_len: int = 4,
        max_pairs_per_prefix_key: int = 200_000, neg_per_pos_cap: int = 15,
        use_tfidf: bool = False, test_chunk_size: int | None = 100_000) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1. Load + validate ----------
    log("Loading training sources...")
    s1_tr = load_source(os.path.join(data_dir, "train", "train_source1.tsv"), "S1-")
    s2_tr = load_source(os.path.join(data_dir, "train", "train_source2.tsv"), "S2-")
    s3_tr = load_source(os.path.join(data_dir, "train", "train_source3.tsv"), "S3-")
    gt = load_ground_truth(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    validate_ground_truth_refs(gt, set(s1_tr.entity_id), set(s2_tr.entity_id), set(s3_tr.entity_id))
    log(f"Train sizes: S1={len(s1_tr)} S2={len(s2_tr)} S3={len(s3_tr)} | "
        f"GT rows={len(gt)} | singleton rate={sum(1 for v in gt.values() if not v)/len(gt):.3f}")

    if sample_s1 is not None and sample_s1 < len(s1_tr):
        log(f"--sample-s1: subsampling to {sample_s1} S1 (matched S2/S3 kept, distractors proportional).")
        rng = np.random.default_rng(seed)
        keep_s1 = set(rng.choice(s1_tr['entity_id'].to_numpy(), size=sample_s1, replace=False))
        s1_tr = s1_tr[s1_tr['entity_id'].isin(keep_s1)].reset_index(drop=True)
        gt = {k: v for k, v in gt.items() if k in keep_s1}
        needed = set().union(*gt.values()) if gt else set()
        s2_extra = set(rng.choice(s2_tr['entity_id'].to_numpy(), size=min(len(s2_tr), sample_s1 * 3), replace=False))
        s3_extra = set(rng.choice(s3_tr['entity_id'].to_numpy(), size=min(len(s3_tr), sample_s1 * 3), replace=False))
        s2_tr = s2_tr[s2_tr['entity_id'].isin(needed | s2_extra)].reset_index(drop=True)
        s3_tr = s3_tr[s3_tr['entity_id'].isin(needed | s3_extra)].reset_index(drop=True)
        log(f"Sampled train: S1={len(s1_tr)} S2={len(s2_tr)} S3={len(s3_tr)}")

    log("Loading test sources...")
    s1_te = load_source(os.path.join(data_dir, "test", "test_source1.tsv"), "S1-")
    s2_te = load_source(os.path.join(data_dir, "test", "test_source2.tsv"), "S2-")
    s3_te = load_source(os.path.join(data_dir, "test", "test_source3.tsv"), "S3-")
    log(f"Test sizes: S1={len(s1_te)} S2={len(s2_te)} S3={len(s3_te)} | "
        f"test countries={sorted(s1_te.country.unique())}")

    # ---------- 2. Normalize ----------
    log("Normalizing text fields...")
    s1_tr, s2_tr, s3_tr = (add_norm_columns(d) for d in (s1_tr, s2_tr, s3_tr))
    s1_te, s2_te, s3_te = (add_norm_columns(d) for d in (s1_te, s2_te, s3_te))

    def block(s1, other):
        if use_tfidf:
            return generate_candidates(s1, other)
        return union_candidates(
            generate_candidates_token(s1, other, max_df=max_df),
            generate_candidates_prefix(s1, other, prefix_len=prefix_len,
                                       max_pairs_per_key=max_pairs_per_prefix_key))

    # ---------- 3. Blocking on TRAIN ----------
    log(f"Blocking train S1 vs S2 (max_df={max_df})...")
    cand_tr_s2 = block(s1_tr, s2_tr)
    log(f"Blocking train S1 vs S3 (max_df={max_df})...")
    cand_tr_s3 = block(s1_tr, s3_tr)
    cand_tr = union_candidates(cand_tr_s2, cand_tr_s3)
    log(f"Train candidate pairs: {sum(len(v) for v in cand_tr.values()):,}")
    recall = candidate_recall(gt, cand_tr)
    log(f"Train candidate recall: {recall:.4f}")
    if recall < 0.90:
        log("WARNING: recall < 0.90 — lower --max-df / --prefix-len before trusting matcher.")

    # ---------- 4. Build labeled pairs (vectorized) ----------
    log("Building pairwise feature frame (vectorized merges)...")
    pairs_tr = pd.concat([build_pair_frame(s1_tr, s2_tr, cand_tr_s2),
                          build_pair_frame(s1_tr, s3_tr, cand_tr_s3)], ignore_index=True)
    feat_df = build_feature_frame_vectorized(pairs_tr)
    labels = np.array([1 if o in gt.get(s, set()) else 0
                       for s, o in zip(feat_df["s1_id"].to_numpy(), feat_df["other_id"].to_numpy())])
    log(f"Pairs before subsample: {len(feat_df):,} | pos: {labels.sum():,}")
    if neg_per_pos_cap > 0:
        feat_df, labels = subsample_negatives(feat_df, labels, neg_per_pos_cap, seed=seed)
        log(f"Pairs after subsample (cap={neg_per_pos_cap}x): {len(feat_df):,} | pos: {labels.sum():,}")

    # ---------- 5. Train grouped-OOF GBDT ----------
    log("Training GBDT with GroupKFold (grouped by S1 entity)...")
    oof_raw, models = train_oof(feat_df, labels, n_splits=n_splits, seed=seed)

    # ---------- 6. Calibrate ----------
    log("Calibrating OOF probabilities (isotonic)...")
    iso = calibrate_oof(oof_raw, labels)
    oof_calibrated = iso.predict(oof_raw)

    # ---------- 7. Tune threshold against macro F0.5 ----------
    log("Tuning decision threshold against macro F0.5...")
    all_s1_ids = s1_tr["entity_id"].tolist()
    best_t, best_score = tune_threshold(feat_df, oof_calibrated, gt, all_s1_ids)
    log(f"Best threshold={best_t:.3f} -> OOF macro F0.5={best_score:.4f}")

    # sanity: all-singleton baseline, for comparison
    empty_preds = {eid: set() for eid in all_s1_ids}
    from metric import macro_f05
    baseline = macro_f05(gt, empty_preds, entity_ids=all_s1_ids)
    log(f"All-singleton baseline macro F0.5 on train: {baseline:.4f} (your real edge over the "
        f"leaderboard is measured above this floor, not from 0)")

    # ---------- 8. Inference: blocking on TEST (chunked to bound peak RAM) ----------
    # The full test join (1.7M S1 x ~10M S2/S3) never fits in RAM at once, and
    # outputs are only written at the end — so a mid-run OOM loses everything.
    # Chunking test S1 keeps peak RAM flat; matches/candidates accumulate incrementally.
    import gc
    all_test_s1_ids = s1_te["entity_id"].tolist()
    matches: Dict[str, Set[str]] = {eid: set() for eid in all_test_s1_ids}
    cand_te: Dict[str, Set[str]] = {}
    chunk = test_chunk_size or len(s1_te)
    n_chunks = (len(s1_te) + chunk - 1) // chunk
    log(f"Test inference in {n_chunks} chunk(s) of ~{chunk:,} S1 (test S1={len(s1_te):,})...")
    for ci in range(n_chunks):
        s1_chunk = s1_te.iloc[ci * chunk:(ci + 1) * chunk]
        t0 = time.time()
        c_s2 = block(s1_chunk, s2_te)
        c_s3 = block(s1_chunk, s3_te)
        pairs_te = pd.concat([build_pair_frame(s1_chunk, s2_te, c_s2),
                              build_pair_frame(s1_chunk, s3_te, c_s3)], ignore_index=True)
        feat_te = build_feature_frame_vectorized(pairs_te)
        if len(feat_te) > 0:
            probs = iso.predict(predict_with_models(models, feat_te[FEATURE_NAMES].to_numpy()))
            keep = probs >= best_t
            for s1_id, other_id in zip(feat_te["s1_id"].to_numpy()[keep],
                                       feat_te["other_id"].to_numpy()[keep]):
                matches[s1_id].add(other_id)
        for k, v in c_s2.items():
            cand_te.setdefault(k, set()).update(v)
        for k, v in c_s3.items():
            cand_te.setdefault(k, set()).update(v)
        log(f"chunk {ci + 1}/{n_chunks}: pairs={len(feat_te):,} "
            f"matched_so_far={sum(1 for v in matches.values() if v):,} ({time.time() - t0:.1f}s)")
        del c_s2, c_s3, pairs_te, feat_te
        gc.collect()

    # ---------- 9. Write outputs ----------
    write_result_tsv(os.path.join(out_dir, "matching_results.tsv"), matches, all_test_s1_ids)
    write_result_tsv(os.path.join(out_dir, "candidate_pairs.tsv"), cand_te, all_test_s1_ids)
    log(f"Wrote matching_results.tsv and candidate_pairs.tsv to {out_dir}")

    n_matched = sum(1 for v in matches.values() if v)
    log(f"Predicted non-singletons: {n_matched}/{len(all_test_s1_ids)} "
        f"({n_matched/len(all_test_s1_ids):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir containing train/ and test/ subfolders")
    ap.add_argument("--out-dir", required=True, help="dir to write matching_results.tsv / candidate_pairs.tsv")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-s1", type=int, default=None)
    ap.add_argument("--max-df", type=int, default=300)
    ap.add_argument("--prefix-len", type=int, default=4)
    ap.add_argument("--max-pairs-per-prefix-key", type=int, default=200_000)
    ap.add_argument("--neg-per-pos-cap", type=int, default=15)
    ap.add_argument("--use-tfidf", action="store_true", help="small-sample TF-IDF blocking only")
    ap.add_argument("--test-chunk-size", type=int, default=100_000,
                    help="test S1 rows per inference chunk; bounds peak RAM (0 = no chunking)")
    ap.add_argument("--validate", action="store_true",
                     help="also run utils/validate_submission.py if found alongside --data-dir")
    args = ap.parse_args()

    run(args.data_dir, args.out_dir, n_splits=args.n_splits, seed=args.seed,
        sample_s1=args.sample_s1, max_df=args.max_df, prefix_len=args.prefix_len,
        max_pairs_per_prefix_key=args.max_pairs_per_prefix_key,
        neg_per_pos_cap=args.neg_per_pos_cap, use_tfidf=args.use_tfidf,
        test_chunk_size=(args.test_chunk_size or None))

    if args.validate:
        validator = os.path.join(args.data_dir, "utils", "validate_submission.py")
        if os.path.exists(validator):
            log("Running official validator...")
            subprocess.run([
                sys.executable, validator,
                "--matching", os.path.join(args.out_dir, "matching_results.tsv"),
                "--candidate", os.path.join(args.out_dir, "candidate_pairs.tsv"),
                "--test-dir", os.path.join(args.data_dir, "test"),
            ], check=False)
        else:
            log(f"Validator not found at {validator}, skipping.")


if __name__ == "__main__":
    main()
