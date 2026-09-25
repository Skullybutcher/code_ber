"""
Pairwise feature engineering between a Source-1 record and a candidate
Source-2/3 record. Kept dependency-light (rapidfuzz + stdlib) so it runs
fast over large candidate sets on Kaggle CPU.
"""
from __future__ import annotations
from typing import Dict, List

from rapidfuzz import fuzz
from normalize import tokens, rare_tokens

FEATURE_NAMES = [
    "name_levenshtein_ratio", "name_token_sort_ratio", "name_partial_ratio",
    "name_jaccard", "name_len_diff", "name_char_bigram_jaccard",
    "addr_levenshtein_ratio", "addr_token_sort_ratio",
    "addr_jaccard", "addr_len_diff",
    "country_match", "country_either_empty",
    "rare_token_overlap", "rare_token_union_size",
    "name_first_token_match",
]


def _char_bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(
    name1: str, addr1: str, country1: str,
    name2: str, addr2: str, country2: str,
) -> List[float]:
    n1_tok, n2_tok = tokens(name1), tokens(name2)
    a1_tok, a2_tok = tokens(addr1), tokens(addr2)
    r1_tok = rare_tokens(name1) | rare_tokens(addr1)
    r2_tok = rare_tokens(name2) | rare_tokens(addr2)

    rare_union = r1_tok | r2_tok
    rare_overlap = len(r1_tok & r2_tok)

    f1_tok = next(iter(sorted(n1_tok)), "")
    f2_tok = next(iter(sorted(n2_tok)), "")

    return [
        fuzz.ratio(name1, name2) / 100.0,
        fuzz.token_sort_ratio(name1, name2) / 100.0,
        fuzz.partial_ratio(name1, name2) / 100.0,
        _jaccard(n1_tok, n2_tok),
        abs(len(name1) - len(name2)),
        _jaccard(_char_bigrams(name1), _char_bigrams(name2)),
        fuzz.ratio(addr1, addr2) / 100.0,
        fuzz.token_sort_ratio(addr1, addr2) / 100.0,
        _jaccard(a1_tok, a2_tok),
        abs(len(addr1) - len(addr2)),
        1.0 if country1 and country2 and country1.lower() == country2.lower() else 0.0,
        1.0 if (not country1 or not country2) else 0.0,
        float(rare_overlap),
        float(len(rare_union)),
        1.0 if f1_tok and f1_tok == f2_tok else 0.0,
    ]


def build_feature_frame(pairs: List[Dict]) -> "pandas.DataFrame":
    """
    pairs: list of dicts each with keys
      s1_id, other_id, name1, addr1, country1, name2, addr2, country2
    Small-sample path. For scale use build_feature_frame_vectorized().
    """
    import pandas as pd
    rows = []
    for p in pairs:
        feats = pair_features(
            p["name1"], p["addr1"], p["country1"],
            p["name2"], p["addr2"], p["country2"],
        )
        rows.append([p["s1_id"], p["other_id"]] + feats)
    return pd.DataFrame(rows, columns=["s1_id", "other_id"] + FEATURE_NAMES)


def build_feature_frame_vectorized(pairs_df: "pandas.DataFrame") -> "pandas.DataFrame":
    """Vectorized-merge friendly: columns s1_id, other_id, name1/addr1/country1, name2/addr2/country2."""
    import numpy as np
    import pandas as pd
    if len(pairs_df) == 0:
        return pd.DataFrame(columns=["s1_id", "other_id"] + FEATURE_NAMES)
    n1 = pairs_df["name1"].fillna("").tolist()
    a1 = pairs_df["addr1"].fillna("").tolist()
    c1 = pairs_df["country1"].fillna("").tolist()
    n2 = pairs_df["name2"].fillna("").tolist()
    a2 = pairs_df["addr2"].fillna("").tolist()
    c2 = pairs_df["country2"].fillna("").tolist()
    feats = [pair_features(x1, y1, z1, x2, y2, z2)
             for x1, y1, z1, x2, y2, z2 in zip(n1, a1, c1, n2, a2, c2)]
    out = pd.DataFrame(np.asarray(feats, dtype=np.float32), columns=FEATURE_NAMES)
    out.insert(0, "s1_id", pairs_df["s1_id"].to_numpy())
    out.insert(1, "other_id", pairs_df["other_id"].to_numpy())
    return out
