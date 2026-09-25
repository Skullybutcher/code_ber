# Business Entity Resolution — Runnable Pipeline

A complete, reproducible pipeline for the Amazon ML Challenge 2026 Business
Entity Resolution task: normalize -> multi-channel block -> pairwise GBDT
match -> calibrate -> threshold-tune against macro F0.5 -> singleton-aware
inference -> `matching_results.tsv` + `candidate_pairs.tsv`.

## Folder layout expected

```
<data-dir>/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
├── test/
│   ├── test_source1.tsv
│   ├── test_source2.tsv
│   └── test_source3.tsv
└── utils/
    └── validate_submission.py   # the organizers' validator, if you have it
```

On Kaggle, `<data-dir>` is wherever you attached your dataset, typically
`/kaggle/input/<your-dataset-name>/student_resource`.

## Setup (Kaggle notebook, no GPU needed for this pipeline)

```bash
pip install -r requirements.txt
```

## Run

1. **Audit first** — always look at the data before spending compute:
   ```bash
   cd src
   python audit.py --data-dir /kaggle/input/<your-dataset>/student_resource
   ```
   This prints singleton rate, match-degree distribution, country mix,
   S2/S3 balance, and the "all-singleton" macro F0.5 floor. That floor
   tells you how much of any leaderboard score is "free."

2. **Run the full pipeline** (blocking -> training -> calibration ->
   threshold tuning -> test inference -> output files):
   ```bash
   python pipeline.py \
       --data-dir /kaggle/input/<your-dataset>/student_resource \
       --out-dir /kaggle/working/output \
       --n-splits 5 \
       --validate
   ```
   `--validate` additionally runs `utils/validate_submission.py` from
   `<data-dir>` against your generated files, if that script is present.

   Console output includes, in order: dataset sizes and singleton rate,
   train candidate recall from blocking (should be as close to 1.0 as
   practical — if it's under ~0.90, widen `top_k_char` / `top_k_word` /
   `rare_min_len` in `blocking.generate_candidates` before trusting
   anything downstream), OOF macro F0.5 at the tuned threshold, and the
   all-singleton baseline for comparison.

3. **Inspect outputs** in `--out-dir`:
   - `matching_results.tsv` — upload this to the leaderboard portal.
   - `candidate_pairs.tsv` — your blocking stage's final candidate set,
     included in the submission zip, not scored directly.

## What each module does

| File | Responsibility |
|---|---|
| `metric.py` | Official-style macro F0.5 (per-entity, singleton-aware) |
| `io_utils.py` | Strict TSV loading, ID-prefix/duplicate validation, output writer |
| `normalize.py` | Legal-suffix stripping, address abbreviation expansion, tokenization |
| `blocking.py` | Char-ngram TF-IDF + word TF-IDF + rare-token blocking, unioned; candidate-recall measurement |
| `features.py` | ~15 pairwise similarity features (name/address fuzzy match, country, rare-token overlap) |
| `model.py` | GroupKFold LightGBM training with per-entity sample weights, isotonic calibration, macro-F0.5-optimal threshold search |
| `pipeline.py` | Orchestrates the above end-to-end and writes both output TSVs |
| `audit.py` | Standalone EDA / dataset-integrity report, run before the pipeline |

## Design choices worth knowing before you tune this

- **Per-entity sample weighting** (`model.make_sample_weights`): each
  training pair is weighted `1 / (# pairs for its S1 entity)` so entities
  with many candidate pairs don't dominate the loss — this aligns training
  with the macro (per-entity, not per-pair) evaluation metric.
- **Threshold is tuned directly against macro F0.5**, not accuracy/AUC,
  because the metric is precision-heavy (β=0.5) and generic classification
  thresholds (e.g. 0.5) are very unlikely to be optimal here.
- **Grouped K-fold by S1 entity** prevents a form of leakage where pairs
  from the same S1 entity land in both train and validation folds.
- **Country is never hard-coded** anywhere in `normalize.py` or
  `features.py` — it's used only as a match/mismatch feature, so France
  (unseen in training) is handled the same way as any other label.
- **No external lookups** are used anywhere in this pipeline — all
  candidate generation and features are derived only from the provided
  TSVs, per the challenge's fair-play rules.

## Known next steps (not yet implemented here)

- Hard-negative mining from the trained model's confident false positives,
  then a retrain pass.
- A small cross-encoder reranker on top-K candidates for the cases the
  GBDT is least confident about (GPU-accelerated, keep ≤8B params and
  MIT/Apache-2.0 licensed per the rules).
- Country-holdout / leave-one-source-out robustness checks before locking
  the final threshold.
- Ensembling across seeds/fold-model sets rather than a single OOF run.

See the accompanying master plan for the full menu of extensions; this
pipeline implements the highest-ROI subset first (Phases 1–3 of the
execution plan) so you have a complete, validated, submittable system
before adding sophistication.
