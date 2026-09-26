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
import gc
import json
import os
import pickle
import subprocess
import sys
import time
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from rapidfuzz import fuzz

# Optional heavy deps
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None

from io_utils import load_source, load_ground_truth, validate_ground_truth_refs, write_result_tsv
from normalize import normalize_name, normalize_address
from blocking import (generate_candidates, generate_candidates_token,
                      generate_candidates_prefix, build_token_index, query_token_index,
                      build_prefix_index, query_prefix_index,
                      union_candidates, candidate_recall)
from features import build_feature_frame_vectorized, FEATURE_NAMES
from model import train_oof, calibrate_oof, tune_threshold, predict_with_models


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class CrossEncoderReranker:
    """Cross-encoder reranker for hard negative pairs. Uses XLM-RoBERTa/DeBERTa for multilingual support."""
    
    def __init__(self, model_name: str = "xlm-roberta-base", threshold: float = 0.92, 
                 device: str = "cpu", max_length: int = 256, batch_size: int = 32):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed. pip install transformers torch")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1
        ).to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.max_length = max_length
        self.batch_size = batch_size
    
    def load_finetuned(self, path: str):
        """Load fine-tuned weights."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        log(f"Loaded fine-tuned cross-encoder from {path}")
    
    def save_finetuned(self, path: str):
        torch.save(self.model.state_dict(), path)
        log(f"Saved fine-tuned cross-encoder to {path}")
    
    def predict_proba(self, name1: List[str], addr1: List[str], 
                      name2: List[str], addr2: List[str]) -> np.ndarray:
        """Return probability of match for each pair."""
        texts1 = [f"{n} [SEP] {a}" for n, a in zip(name1, addr1)]
        texts2 = [f"{n} [SEP] {a}" for n, a in zip(name2, addr2)]
        
        all_probs = []
        for i in range(0, len(texts1), self.batch_size):
            batch1 = texts1[i:i+self.batch_size]
            batch2 = texts2[i:i+self.batch_size]
            enc = self.tokenizer(
                batch1, batch2, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs if probs.ndim > 0 else [probs])
        return np.array(all_probs)
    
    def predict(self, name1: List[str], addr1: List[str], 
                name2: List[str], addr2: List[str]) -> np.ndarray:
        """Return binary predictions using threshold."""
        return (self.predict_proba(name1, addr1, name2, addr2) >= self.threshold).astype(int)
    
    def finetune(self, name1: List[str], addr1: List[str],
                 name2: List[str], addr2: List[str], labels: np.ndarray,
                 epochs: int = 3, lr: float = 2e-5, val_split: float = 0.1):
        """Fine-tune on hard negatives."""
        from torch.utils.data import DataLoader, TensorDataset
        import torch.nn as nn
        
        texts1 = [f"{n} [SEP] {a}" for n, a in zip(name1, addr1)]
        texts2 = [f"{n} [SEP] {a}" for n, a in zip(name2, addr2)]
        
        enc = self.tokenizer(
            texts1, texts2, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        
        dataset = TensorDataset(
            enc["input_ids"], enc["attention_mask"], 
            torch.tensor(labels, dtype=torch.float32)
        )
        
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels_batch = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            log(f"  Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels_batch = [b.to(self.device) for b in batch]
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                    val_loss += nn.BCEWithLogitsLoss()(logits, labels_batch).item()
            log(f"  Val Loss: {val_loss/len(val_loader):.4f}")
            self.model.train()


class FaissDenseRetriever:
    """Dense retrieval blocking using FAISS and bi-encoder embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", top_k: int = 50):
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss not installed. pip install faiss-cpu")
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not installed")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.top_k = top_k
        self.index = None
        self.other_ids = None
    
    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Encode texts to normalized embeddings."""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, 
                                max_length=128, return_tensors="pt").to(self.device)
            outputs = self.model(**enc)
            # Mean pooling
            emb = outputs.last_hidden_state * enc["attention_mask"].unsqueeze(-1)
            emb = emb.sum(dim=1) / enc["attention_mask"].sum(dim=1, keepdim=True)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)
    
    def build_index(self, other_dfs: List[pd.DataFrame]):
        """Build FAISS index from one or more other side DataFrames."""
        all_texts = []
        all_ids = []
        for df in other_dfs:
            texts = (df["_norm_name"].fillna("") + " " + df["_norm_addr"].fillna("")).tolist()
            all_texts.extend(texts)
            all_ids.extend(df["entity_id"].tolist())
        
        log(f"Encoding {len(all_texts)} records for FAISS index...")
        embs = self.encode(all_texts)
        
        # Build IVF index for large datasets
        d = embs.shape[1]
        nlist = min(4096, max(1, int(len(embs) ** 0.5)))
        quantizer = faiss.IndexFlatIP(d)
        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        self.index.train(embs)
        self.index.add(embs)
        self.index.nprobe = min(32, nlist)
        self.other_ids = np.array(all_ids)
        log(f"FAISS index built with {self.index.ntotal} vectors, nlist={nlist}")
    
    def search(self, s1_chunk: pd.DataFrame) -> Dict[str, Set[str]]:
        """Search for top-K candidates for S1 chunk."""
        texts = (s1_chunk["_norm_name"].fillna("") + " " + s1_chunk["_norm_addr"].fillna("")).tolist()
        q_embs = self.encode(texts)
        
        scores, idx = self.index.search(q_embs, self.top_k)
        
        candidates = {}
        s1_ids = s1_chunk["entity_id"].to_numpy()
        for i, s1_id in enumerate(s1_ids):
            cand = set()
            for j, score in zip(idx[i], scores[i]):
                if score > 0.3:  # Minimum similarity threshold
                    cand.add(self.other_ids[j])
            candidates[s1_id] = cand
        return candidates


def generate_candidates_faiss(s1_df: pd.DataFrame, other_df: pd.DataFrame, 
                              retriever: FaissDenseRetriever) -> Dict[str, Set[str]]:
    """Wrapper for FAISS dense retrieval blocking."""
    return retriever.search(s1_df)


def mine_hard_negatives(feat_df: pd.DataFrame, labels: np.ndarray, 
                        calibrated_probs: np.ndarray,
                        top_k: int = 5000, margin: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mine hard negatives: false positives with high probability (confident mistakes)
    and false negatives with low probability (missed matches).
    Returns additional hard negative indices and their labels.
    """
    fp_mask = (labels == 0) & (calibrated_probs >= 0.5)  # Confident false positives
    fn_mask = (labels == 1) & (calibrated_probs < 0.5)   # Missed true matches
    
    fp_indices = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]
    
    # Sort FPs by probability descending (most confident mistakes first)
    if len(fp_indices) > 0:
        fp_probs = calibrated_probs[fp_indices]
        fp_order = np.argsort(-fp_probs)
        fp_indices = fp_indices[fp_order[:min(top_k, len(fp_indices))]]
    
    # Sort FNs by probability ascending (most confident misses first)
    if len(fn_indices) > 0:
        fn_probs = calibrated_probs[fn_indices]
        fn_order = np.argsort(fn_probs)
        fn_indices = fn_indices[fn_order[:min(top_k, len(fn_indices))]]
    
    hard_neg_idx = np.concatenate([fp_indices, fn_indices])
    hard_neg_labels = labels[hard_neg_idx]
    
    return hard_neg_idx, hard_neg_labels


def consensus_predict(tfidf_probs: np.ndarray, ce_probs: np.ndarray, 
                      threshold_tfidf: float = 0.65, threshold_ce: float = 0.92,
                      mode: str = "and") -> np.ndarray:
    """Consensus prediction: require both TF-IDF and Cross-Encoder to agree."""
    tfidf_pred = tfidf_probs >= threshold_tfidf
    ce_pred = ce_probs >= threshold_ce
    
    if mode == "and":
        return (tfidf_pred & ce_pred).astype(int)
    elif mode == "or":
        return (tfidf_pred | ce_pred).astype(int)
    elif mode == "weighted":
        # Weighted ensemble: 0.6 * ce + 0.4 * tfidf
        combined = 0.6 * ce_probs + 0.4 * tfidf_probs
        return (combined >= 0.5).astype(int)
    else:
        raise ValueError(f"Unknown mode: {mode}")
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


def _cache_key(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.basename(path)}.{st.st_size}.{int(st.st_mtime)}"


def load_or_build_frames(data_dir: str, cache_dir: str | None) -> tuple:
    """Load all six sources normalized, reusing a parquet cache when valid.

    TSV parse + normalize of 21M rows costs ~150-180s every run. The cache
    stores post-normalization frames keyed on (filename, size, mtime).
    """
    specs = [("train", "train_source1.tsv", "S1-"), ("train", "train_source2.tsv", "S2-"),
             ("train", "train_source3.tsv", "S3-"), ("test", "test_source1.tsv", "S1-"),
             ("test", "test_source2.tsv", "S2-"), ("test", "test_source3.tsv", "S3-")]
    manifest_path = os.path.join(cache_dir, "manifest.json") if cache_dir else None
    manifest: dict = {}
    if manifest_path and os.path.exists(manifest_path):
        import json
        with open(manifest_path) as f:
            manifest = json.load(f)
    frames, new_manifest = [], {}
    for split, fname, prefix in specs:
        path = os.path.join(data_dir, split, fname)
        key = _cache_key(path)
        cached = os.path.join(cache_dir, f"{split}_{fname}.parquet") if cache_dir else None
        if cached and manifest.get(fname) == key and os.path.exists(cached):
            log(f"Cache hit: {fname}")
            frames.append(pd.read_parquet(cached))
        else:
            log(f"Cache miss: {fname} (loading + normalizing)...")
            df = add_norm_columns(load_source(path, prefix))
            frames.append(df)
            if cached:
                df.to_parquet(cached, index=False)
        new_manifest[fname] = key
    if manifest_path:
        import json
        with open(manifest_path, "w") as f:
            json.dump(new_manifest, f)
    return tuple(frames)


def save_artifacts(model_dir: str, models, iso, threshold: float, oof_score: float) -> None:
    os.makedirs(model_dir, exist_ok=True)
    import json, pickle
    for i, m in enumerate(models):
        m.save_model(os.path.join(model_dir, f"model_{i}.txt"))
    with open(os.path.join(model_dir, "calibrator.pkl"), "wb") as f:
        pickle.dump(iso, f)
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump({"threshold": threshold, "oof_score": oof_score,
                   "n_models": len(models), "features": FEATURE_NAMES}, f)
    log(f"Saved {len(models)} models + calibrator + threshold={threshold:.3f} to {model_dir}")


def load_artifacts(model_dir: str):
    import json, pickle
    import lightgbm as lgb
    with open(os.path.join(model_dir, "meta.json")) as f:
        meta = json.load(f)
    models = [lgb.Booster(model_file=os.path.join(model_dir, f"model_{i}.txt"))
              for i in range(meta["n_models"])]
    with open(os.path.join(model_dir, "calibrator.pkl"), "rb") as f:
        iso = pickle.load(f)
    log(f"Loaded {len(models)} models (OOF={meta['oof_score']:.4f}) from {model_dir}; "
        f"skipping train blocking/training.")
    return models, iso, float(meta["threshold"])


def run(data_dir: str, out_dir: str, n_splits: int = 5, seed: int = 42,
        sample_s1: int | None = None, max_df: int = 300, prefix_len: int = 4,
        max_pairs_per_prefix_key: int = 200_000, neg_per_pos_cap: int = 15,
        use_tfidf: bool = False, test_chunk_size: int | None = 100_000,
        min_len: int = 4, cache_dir: str | None = None,
        save_model_dir: str | None = None, load_model_dir: str | None = None,
        use_cross_encoder: bool = False, cross_encoder_model: str = "xlm-roberta-base",
        cross_encoder_threshold: float = 0.92, use_faiss: bool = False,
        faiss_top_k: int = 50, faiss_device: str = "cpu") -> None:
    os.makedirs(out_dir, exist_ok=True)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # ---------- 1. Load + validate (parquet cache when valid) ----------
    log("Loading sources...")
    s1_tr, s2_tr, s3_tr, s1_te, s2_te, s3_te = load_or_build_frames(data_dir, cache_dir)
    gt = load_ground_truth(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    validate_ground_truth_refs(gt, set(s1_tr.entity_id), set(s2_tr.entity_id), set(s3_tr.entity_id))
    # ---------- 2. Normalize (already applied by load_or_build_frames) ----------
    log(f"Train sizes: S1={len(s1_tr)} S2={len(s2_tr)} S3={len(s3_tr)} | "
        f"GT rows={len(gt)} | singleton rate={sum(1 for v in gt.values() if not v)/len(gt):.3f}")
    log(f"Test sizes: S1={len(s1_te)} S2={len(s2_te)} S3={len(s3_te)} | "
        f"test countries={sorted(s1_te.country.unique())}")

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

    def block(s1, other):
        if use_tfidf:
            return generate_candidates(s1, other)
        return union_candidates(
            generate_candidates_token(s1, other, min_len=min_len, max_df=max_df),
            generate_candidates_prefix(s1, other, prefix_len=prefix_len,
                                       max_pairs_per_key=max_pairs_per_prefix_key))

    if load_model_dir:
        models, iso, best_t = load_artifacts(load_model_dir)
        best_score = float("nan")
    else:
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

    if save_model_dir:
        save_artifacts(save_model_dir, models, iso, best_t, best_score)

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
    use_index = not use_tfidf and n_chunks > 1
    
    # Optional: FAISS dense retrieval index (built once, reused across chunks)
    faiss_retriever = None
    if use_faiss and FAISS_AVAILABLE:
        log("Building FAISS dense retrieval index...")
        t_idx = time.time()
        faiss_retriever = FaissDenseRetriever(top_k=faiss_top_k, device=faiss_device)
        faiss_retriever.build_index([s2_te, s3_te])  # combined index for both S2 and S3
        log(f"FAISS index ready ({time.time() - t_idx:.1f}s).")
    
    # Optional: Cross-encoder reranker (loaded once, reused across chunks)
    cross_encoder = None
    if use_cross_encoder and TRANSFORMERS_AVAILABLE:
        log("Loading cross-encoder reranker...")
        t_ce = time.time()
        cross_encoder = CrossEncoderReranker(
            model_name=cross_encoder_model, threshold=cross_encoder_threshold,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        log(f"Cross-encoder loaded ({time.time() - t_ce:.1f}s).")
    
    if use_index:
        log("Precomputing other-side blocking indices once (reused by all chunks)...")
        t_idx = time.time()
        tok_idx_s2 = build_token_index(s2_te, min_len=min_len, max_df=max_df)
        tok_idx_s3 = build_token_index(s3_te, min_len=min_len, max_df=max_df)
        pfx_idx_s2, pfx_cnt_s2 = build_prefix_index(s2_te, prefix_len=prefix_len)
        pfx_idx_s3, pfx_cnt_s3 = build_prefix_index(s3_te, prefix_len=prefix_len)
        log(f"Indices ready ({time.time() - t_idx:.1f}s).")
    else:
        tok_idx_s2 = tok_idx_s3 = pfx_idx_s2 = pfx_idx_s3 = None
        pfx_cnt_s2 = pfx_cnt_s3 = None
    
    for ci in range(n_chunks):
        s1_chunk = s1_te.iloc[ci * chunk:(ci + 1) * chunk]
        t0 = time.time()
        if use_index:
            c_s2 = union_candidates(
                query_token_index(s1_chunk, tok_idx_s2, min_len=min_len, max_df=max_df),
                query_prefix_index(s1_chunk, pfx_idx_s2, pfx_cnt_s2, prefix_len=prefix_len,
                                   max_pairs_per_key=max_pairs_per_prefix_key))
            c_s3 = union_candidates(
                query_token_index(s1_chunk, tok_idx_s3, min_len=min_len, max_df=max_df),
                query_prefix_index(s1_chunk, pfx_idx_s3, pfx_cnt_s3, prefix_len=prefix_len,
                                   max_pairs_per_key=max_pairs_per_prefix_key))
        else:
            c_s2 = block(s1_chunk, s2_te)
            c_s3 = block(s1_chunk, s3_te)
        
        # FAISS dense retrieval candidates (union with token/prefix)
        if faiss_retriever is not None:
            faiss_cand = faiss_retriever.search(s1_chunk)
            c_s2 = union_candidates(c_s2, faiss_cand)
            c_s3 = union_candidates(c_s3, faiss_cand)
        
        pairs_te = pd.concat([build_pair_frame(s1_chunk, s2_te, c_s2),
                              build_pair_frame(s1_chunk, s3_te, c_s3)], ignore_index=True)
        feat_te = build_feature_frame_vectorized(pairs_te)
        
        if len(feat_te) > 0:
            # LightGBM probabilities
            lgb_probs = iso.predict(predict_with_models(models, feat_te[FEATURE_NAMES].to_numpy()))
            
            # Cross-encoder reranking for consensus
            if cross_encoder is not None:
                log(f"  Cross-encoder reranking {len(feat_te):,} pairs...")
                ce_probs = cross_encoder.predict_proba(
                    feat_te["name1"].fillna("").tolist(),
                    feat_te["addr1"].fillna("").tolist(),
                    feat_te["name2"].fillna("").tolist(),
                    feat_te["addr2"].fillna("").tolist()
                )
                # Consensus: both LightGBM and Cross-Encoder must agree (AND)
                keep = consensus_predict(lgb_probs, ce_probs, 
                                         threshold_tfidf=best_t, threshold_ce=cross_encoder_threshold,
                                         mode="and")
            else:
                keep = lgb_probs >= best_t
            
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
    # Advanced options
    ap.add_argument("--min-len", type=int, default=4,
                    help="min token length for token blocking (higher = fewer candidates)")
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="directory for parquet cache of normalized frames")
    ap.add_argument("--save-model-dir", type=str, default=None,
                    help="directory to save trained model artifacts")
    ap.add_argument("--load-model-dir", type=str, default=None,
                    help="directory to load trained model artifacts (skips training)")
    ap.add_argument("--use-faiss", action="store_true",
                    help="enable FAISS dense retrieval blocking (requires faiss-cpu)")
    ap.add_argument("--faiss-top-k", type=int, default=50,
                    help="top-K candidates from FAISS dense retrieval per S1 entity")
    ap.add_argument("--faiss-device", type=str, default="cpu", choices=["cpu", "cuda"],
                    help="device for FAISS (cpu or cuda)")
    ap.add_argument("--use-cross-encoder", action="store_true",
                    help="enable cross-encoder reranker for consensus (requires transformers)")
    ap.add_argument("--cross-encoder-model", type=str, default="xlm-roberta-base",
                    help="HF model name for cross-encoder (e.g., xlm-roberta-base, deberta-v3-base)")
    ap.add_argument("--cross-encoder-threshold", type=float, default=0.92,
                    help="probability threshold for cross-encoder (consensus AND with LightGBM)")
    ap.add_argument("--consensus-mode", type=str, default="and", choices=["and", "or", "weighted"],
                    help="consensus mode for TF-IDF + cross-encoder ensemble")
    args = ap.parse_args()

    run(args.data_dir, args.out_dir, n_splits=args.n_splits, seed=args.seed,
        sample_s1=args.sample_s1, max_df=args.max_df, prefix_len=args.prefix_len,
        max_pairs_per_prefix_key=args.max_pairs_per_prefix_key,
        neg_per_pos_cap=args.neg_per_pos_cap, use_tfidf=args.use_tfidf,
        test_chunk_size=(args.test_chunk_size or None),
        min_len=args.min_len, cache_dir=args.cache_dir,
        save_model_dir=args.save_model_dir, load_model_dir=args.load_model_dir,
        use_faiss=args.use_faiss, faiss_top_k=args.faiss_top_k,
        faiss_device=args.faiss_device, use_cross_encoder=args.use_cross_encoder,
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_threshold=args.cross_encoder_threshold)

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
