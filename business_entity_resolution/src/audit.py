"""
Run this FIRST, before the main pipeline. Prints dataset integrity checks
and ground-truth graph statistics so you know what you're working with
(singleton rate, match multiplicity, country mix, S2/S3 balance) before
spending compute on blocking/training.

Usage:
    python audit.py --data-dir /kaggle/input/<your-dataset>
"""
from __future__ import annotations
import argparse
import os
from collections import Counter

from io_utils import load_source, load_ground_truth, validate_ground_truth_refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    d = args.data_dir

    s1 = load_source(os.path.join(d, "train", "train_source1.tsv"), "S1-")
    s2 = load_source(os.path.join(d, "train", "train_source2.tsv"), "S2-")
    s3 = load_source(os.path.join(d, "train", "train_source3.tsv"), "S3-")
    gt = load_ground_truth(os.path.join(d, "train", "train_ground_truth.tsv"))
    validate_ground_truth_refs(gt, set(s1.entity_id), set(s2.entity_id), set(s3.entity_id))

    print("=== Sizes ===")
    print(f"S1: {len(s1)}  S2: {len(s2)}  S3: {len(s3)}  GT rows: {len(gt)}")

    print("\n=== Country distribution (train) ===")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        print(name, dict(Counter(df.country)))

    print("\n=== Missingness (train) ===")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        for col in ["business_name", "business_address", "country"]:
            n_empty = (df[col].str.strip() == "").sum()
            if n_empty:
                print(f"{name}.{col}: {n_empty} empty ({n_empty/len(df):.3%})")

    degrees = [len(v) for v in gt.values()]
    deg_counts = Counter(degrees)
    print("\n=== Ground-truth degree distribution ===")
    for deg in sorted(deg_counts):
        print(f"  degree={deg}: {deg_counts[deg]} entities ({deg_counts[deg]/len(gt):.3%})")
    print(f"Singleton rate: {deg_counts.get(0, 0) / len(gt):.4f}")

    s2_only = sum(1 for v in gt.values() if v and all(x.startswith("S2-") for x in v))
    s3_only = sum(1 for v in gt.values() if v and all(x.startswith("S3-") for x in v))
    both = sum(1 for v in gt.values() if any(x.startswith("S2-") for x in v) and any(x.startswith("S3-") for x in v))
    print(f"S2-only matches: {s2_only}  S3-only matches: {s3_only}  both: {both}")

    # check if any S2/S3 record is claimed by more than one S1 (many-to-one on the other side)
    other_owner = Counter()
    for v in gt.values():
        for other in v:
            other_owner[other] += 1
    multi_owned = {k: c for k, c in other_owner.items() if c > 1}
    print(f"\nS2/S3 records matched to >1 Source-1 entity: {len(multi_owned)}"
          f" {'(sample: ' + str(list(multi_owned.items())[:5]) + ')' if multi_owned else ''}")
    print("If this is non-zero, do NOT assume a global one-to-one constraint downstream.")

    # all-singleton floor
    from metric import macro_f05
    all_s1_ids = s1["entity_id"].tolist()
    empty_preds = {eid: set() for eid in all_s1_ids}
    baseline = macro_f05(gt, empty_preds, entity_ids=all_s1_ids)
    print(f"\nAll-singleton baseline macro F0.5: {baseline:.4f}")
    print("(This is your free floor. The leaderboard's real difficulty is the gap above this number.)")


if __name__ == "__main__":
    main()
