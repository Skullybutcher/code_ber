"""
Strict TSV loading + dataset integrity checks (PS-01 from the master plan,
compressed). Fails loudly on structural problems rather than silently
producing a corrupted pipeline.
"""
from __future__ import annotations
import pandas as pd
from typing import Dict, Set

REQUIRED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: str, expected_prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}, got {list(df.columns)}")

    bad_prefix = df.loc[~df["entity_id"].str.startswith(expected_prefix), "entity_id"]
    if len(bad_prefix) > 0:
        raise ValueError(
            f"{path}: {len(bad_prefix)} entity_id values do not start with "
            f"'{expected_prefix}' (e.g. {bad_prefix.head(3).tolist()})"
        )

    dup = df["entity_id"][df["entity_id"].duplicated()]
    if len(dup) > 0:
        raise ValueError(f"{path}: {len(dup)} duplicate entity_id values, e.g. {dup.head(3).tolist()}")

    for col in REQUIRED_SOURCE_COLS:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df.reset_index(drop=True)


def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    required = ["source1_entity_id", "matched_entity_ids"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    dup = df["source1_entity_id"][df["source1_entity_id"].duplicated()]
    if len(dup) > 0:
        raise ValueError(f"{path}: duplicate source1_entity_id rows: {dup.head(3).tolist()}")

    gt: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        raw = (row["matched_entity_ids"] or "").strip()
        ids = set(x.strip() for x in raw.split(",") if x.strip()) if raw else set()
        gt[row["source1_entity_id"]] = ids
    return gt


def validate_ground_truth_refs(
    gt: Dict[str, Set[str]],
    s1_ids: Set[str],
    s2_ids: Set[str],
    s3_ids: Set[str],
) -> None:
    missing_s1 = set(gt.keys()) - s1_ids
    if missing_s1:
        raise ValueError(f"ground truth references {len(missing_s1)} unknown Source-1 ids, e.g. {list(missing_s1)[:3]}")
    all_valid_targets = s2_ids | s3_ids
    bad_targets = set()
    for matched in gt.values():
        bad_targets |= (matched - all_valid_targets)
    if bad_targets:
        raise ValueError(f"ground truth references {len(bad_targets)} unknown S2/S3 ids, e.g. {list(bad_targets)[:3]}")
    missing_gt_rows = s1_ids - set(gt.keys())
    if missing_gt_rows:
        raise ValueError(f"{len(missing_gt_rows)} Source-1 training entities have no ground-truth row")


def write_result_tsv(path: str, predictions: Dict[str, Set[str]], s1_order) -> None:
    """Write matching_results.tsv / candidate_pairs.tsv in the required format."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n" if "matching" in path
                else "source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_order:
            ids = predictions.get(eid, set())
            f.write(f"{eid}\t{','.join(sorted(ids))}\n")
