"""
Multi-channel blocking: char n-gram TF-IDF top-K, word TF-IDF top-K, and a
rare-token inverted index, unioned together. Optimized for correctness and
clarity over raw speed; for ~1GB of data this runs source-by-source with
sparse matrices which keeps memory manageable.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from normalize import normalize_name, normalize_address, rare_tokens


def _blocking_text(row_name: str, row_addr: str) -> str:
    # Name weighted more heavily than address by simple repetition.
    return f"{row_name} {row_name} {row_addr}"


def _topk_neighbors(query_texts: List[str], ref_texts: List[str], analyzer: str,
                     ngram_range, top_k: int) -> List[List[int]]:
    if len(ref_texts) == 0 or len(query_texts) == 0:
        return [[] for _ in query_texts]
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=1, sublinear_tf=True)
    ref_mat = vec.fit_transform(ref_texts)
    q_mat = vec.transform(query_texts)
    k = min(top_k, ref_mat.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(ref_mat)
    _, idx = nn.kneighbors(q_mat)
    return idx.tolist()


def _rare_token_candidates(s1_df, ref_df, min_len: int = 4) -> Dict[int, Set[int]]:
    """Inverted index on rare (long) tokens shared between name+address."""
    inv = defaultdict(list)
    for j, (name, addr) in enumerate(zip(ref_df["_norm_name"], ref_df["_norm_addr"])):
        for tok in rare_tokens(name, min_len) | rare_tokens(addr, min_len):
            inv[tok].append(j)

    out: Dict[int, Set[int]] = defaultdict(set)
    for i, (name, addr) in enumerate(zip(s1_df["_norm_name"], s1_df["_norm_addr"])):
        toks = rare_tokens(name, min_len) | rare_tokens(addr, min_len)
        for tok in toks:
            for j in inv.get(tok, ()):
                out[i].add(j)
    return out


def generate_candidates(s1_df, other_df, top_k_char: int = 15, top_k_word: int = 15,
                         rare_min_len: int = 4) -> Dict[str, Set[str]]:
    """
    Returns {s1_entity_id: set(other_entity_id)} candidate pairs for ONE
    other source (call twice, once for Source2 and once for Source3, then
    union the results per S1 entity).
    """
    for df in (s1_df, other_df):
        if "_norm_name" not in df.columns:
            df["_norm_name"] = df["business_name"].map(normalize_name)
            df["_norm_addr"] = df["business_address"].map(normalize_address)

    s1_text = [_blocking_text(n, a) for n, a in zip(s1_df["_norm_name"], s1_df["_norm_addr"])]
    other_text = [_blocking_text(n, a) for n, a in zip(other_df["_norm_name"], other_df["_norm_addr"])]

    char_idx = _topk_neighbors(s1_text, other_text, analyzer="char_wb", ngram_range=(2, 4), top_k=top_k_char)
    word_idx = _topk_neighbors(s1_text, other_text, analyzer="word", ngram_range=(1, 2), top_k=top_k_word)
    rare_idx = _rare_token_candidates(s1_df, other_df, min_len=rare_min_len)

    other_ids = other_df["entity_id"].tolist()
    s1_ids = s1_df["entity_id"].tolist()

    result: Dict[str, Set[str]] = {}
    for i, s1_id in enumerate(s1_ids):
        cand = set()
        for j in char_idx[i]:
            cand.add(other_ids[j])
        for j in word_idx[i]:
            cand.add(other_ids[j])
        for j in rare_idx.get(i, ()):
            cand.add(other_ids[j])
        result[s1_id] = cand
    return result


def union_candidates(*candidate_dicts: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for d in candidate_dicts:
        for k, v in d.items():
            out[k] |= v
    return dict(out)


def _token_frame(df, min_len: int = 4):
    """Vectorized (entity_id, token) posting list from name+address tokens."""
    import pandas as pd
    tmp = df[["entity_id"]].copy()
    tmp["token"] = (df["_norm_name"].fillna("") + " " + df["_norm_addr"].fillna("")).str.split()
    tmp = tmp.explode("token")
    tmp = tmp[tmp["token"].notna() & (tmp["token"].str.len() >= min_len)]
    return tmp.drop_duplicates()


def _cap_document_frequency(token_frame, max_df: int):
    """Drop tokens in more than max_df records on this side (join blowup guard)."""
    counts = token_frame["token"].value_counts()
    keep = set(counts[counts <= max_df].index)
    return token_frame[token_frame["token"].isin(keep)]


def build_token_index(other_df, min_len: int = 4, max_df: int = 300):
    """Precompute the capped (entity_id_other, token) posting list for one other side.

    Build ONCE per other source and reuse across test chunks — rebuilding this
    25M-row frame per chunk is what made chunked inference crawl.
    """
    if "_norm_name" not in other_df.columns:
        other_df["_norm_name"] = other_df["business_name"].map(normalize_name)
        other_df["_norm_addr"] = other_df["business_address"].map(normalize_address)
    tok = _cap_document_frequency(_token_frame(other_df, min_len), max_df)
    return tok.rename(columns={"entity_id": "entity_id_other"})


def query_token_index(s1_df, other_index, min_len: int = 4, max_df: int = 300) -> Dict[str, Set[str]]:
    """Token blocking of one S1 chunk against a prebuilt other-side index."""
    if "_norm_name" not in s1_df.columns:
        s1_df["_norm_name"] = s1_df["business_name"].map(normalize_name)
        s1_df["_norm_addr"] = s1_df["business_address"].map(normalize_address)
    s1_tok = _cap_document_frequency(_token_frame(s1_df, min_len), max_df)
    s1_tok = s1_tok.rename(columns={"entity_id": "entity_id_s1"})
    if len(s1_tok) == 0 or len(other_index) == 0:
        return {}
    merged = s1_tok.merge(other_index, on="token")
    pairs = merged[["entity_id_s1", "entity_id_other"]].drop_duplicates()
    return pairs.groupby("entity_id_s1")["entity_id_other"].apply(set).to_dict()


def build_prefix_index(other_df, prefix_len: int = 4):
    """Precompute the (entity_id_other, key) frame + per-key counts for one other side."""
    if "_norm_name" not in other_df.columns:
        other_df["_norm_name"] = other_df["business_name"].map(normalize_name)
        other_df["_norm_addr"] = other_df["business_address"].map(normalize_address)
    other_key = other_df[["entity_id"]].copy()
    other_key["key"] = other_df["country"].str.lower().fillna("") + "|" + other_df["_norm_name"].str[:prefix_len]
    other_key = other_key.rename(columns={"entity_id": "entity_id_other"})
    return other_key, other_key["key"].value_counts()


def query_prefix_index(s1_df, other_key, other_counts, prefix_len: int = 4,
                       max_pairs_per_key: int = 200_000) -> Dict[str, Set[str]]:
    """Prefix blocking of one S1 chunk against a prebuilt other-side index."""
    if "_norm_name" not in s1_df.columns:
        s1_df["_norm_name"] = s1_df["business_name"].map(normalize_name)
        s1_df["_norm_addr"] = s1_df["business_address"].map(normalize_address)
    s1_key = s1_df[["entity_id"]].copy()
    s1_key["key"] = s1_df["country"].str.lower().fillna("") + "|" + s1_df["_norm_name"].str[:prefix_len]
    s1_key = s1_key.rename(columns={"entity_id": "entity_id_s1"})
    s1_counts = s1_key["key"].value_counts()
    common = set(s1_counts.index) & set(other_counts.index)
    safe = {k for k in common if s1_counts[k] * other_counts[k] <= max_pairs_per_key}
    s1_key = s1_key[s1_key["key"].isin(safe)]
    other_sub = other_key[other_key["key"].isin(safe)]
    if len(s1_key) == 0 or len(other_sub) == 0:
        return {}
    merged = s1_key.merge(other_sub, on="key")
    pairs = merged[["entity_id_s1", "entity_id_other"]].drop_duplicates()
    return pairs.groupby("entity_id_s1")["entity_id_other"].apply(set).to_dict()


def generate_candidates_token(s1_df, other_df, min_len: int = 4, max_df: int = 300) -> Dict[str, Set[str]]:
    """Scalable token blocking via vectorized merge. Default for full scale."""
    for df in (s1_df, other_df):
        if "_norm_name" not in df.columns:
            df["_norm_name"] = df["business_name"].map(normalize_name)
            df["_norm_addr"] = df["business_address"].map(normalize_address)
    s1_tok = _cap_document_frequency(_token_frame(s1_df, min_len), max_df)
    other_tok = _cap_document_frequency(_token_frame(other_df, min_len), max_df)
    if len(s1_tok) == 0 or len(other_tok) == 0:
        return {}
    merged = s1_tok.merge(other_tok, on="token", suffixes=("_s1", "_other"))
    pairs = merged[["entity_id_s1", "entity_id_other"]].drop_duplicates()
    return pairs.groupby("entity_id_s1")["entity_id_other"].apply(set).to_dict()


def generate_candidates_prefix(s1_df, other_df, prefix_len: int = 4,
                               max_pairs_per_key: int = 200_000) -> Dict[str, Set[str]]:
    """Cheap (country, name-prefix) channel. Caps on count_s1*count_other per bucket."""
    for df in (s1_df, other_df):
        if "_norm_name" not in df.columns:
            df["_norm_name"] = df["business_name"].map(normalize_name)
            df["_norm_addr"] = df["business_address"].map(normalize_address)
    s1_key = s1_df[["entity_id"]].copy()
    s1_key["key"] = s1_df["country"].str.lower().fillna("") + "|" + s1_df["_norm_name"].str[:prefix_len]
    other_key = other_df[["entity_id"]].copy()
    other_key["key"] = other_df["country"].str.lower().fillna("") + "|" + other_df["_norm_name"].str[:prefix_len]
    s1_counts = s1_key["key"].value_counts()
    other_counts = other_key["key"].value_counts()
    common = set(s1_counts.index) & set(other_counts.index)
    safe = {k for k in common if s1_counts[k] * other_counts[k] <= max_pairs_per_key}
    s1_key = s1_key[s1_key["key"].isin(safe)]
    other_key = other_key[other_key["key"].isin(safe)]
    if len(s1_key) == 0 or len(other_key) == 0:
        return {}
    merged = s1_key.merge(other_key, on="key", suffixes=("_s1", "_other"))
    pairs = merged[["entity_id_s1", "entity_id_other"]].drop_duplicates()
    return pairs.groupby("entity_id_s1")["entity_id_other"].apply(set).to_dict()


def candidate_recall(gt: Dict[str, Set[str]], candidates: Dict[str, Set[str]]) -> float:
    """Fraction of true positive links that survive blocking (upper bound on recall)."""
    total, found = 0, 0
    for s1_id, true_matches in gt.items():
        if not true_matches:
            continue
        cand = candidates.get(s1_id, set())
        total += len(true_matches)
        found += len(true_matches & cand)
    return found / total if total > 0 else 1.0
