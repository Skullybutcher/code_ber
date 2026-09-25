"""CPU blocking benchmark on train sample. Run: python bench_blocking.py --data-dir <data> --n 5000 --max-df 300"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import load_source, load_ground_truth
from normalize import normalize_name, normalize_address
from blocking import generate_candidates_token, generate_candidates_prefix, union_candidates, candidate_recall
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', default='dataset')
ap.add_argument('--n', type=int, default=5000)
ap.add_argument('--max-df', type=int, default=300)
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()
t0 = time.time()
s1 = load_source(os.path.join(a.data_dir, 'train', 'train_source1.tsv'), 'S1-')
s2 = load_source(os.path.join(a.data_dir, 'train', 'train_source2.tsv'), 'S2-')
s3 = load_source(os.path.join(a.data_dir, 'train', 'train_source3.tsv'), 'S3-')
gt = load_ground_truth(os.path.join(a.data_dir, 'train', 'train_ground_truth.tsv'))
print(f"load {time.time()-t0:.1f}s S1={len(s1)} S2={len(s2)} S3={len(s3)}", flush=True)
rng = np.random.default_rng(a.seed)
keep = set(rng.choice(s1['entity_id'].to_numpy(), size=min(a.n, len(s1)), replace=False))
s1s = s1[s1['entity_id'].isin(keep)].reset_index(drop=True)
gts = {k: v for k, v in gt.items() if k in keep}
need = set().union(*gts.values()) if gts else set()
s2s = s2[s2['entity_id'].isin(need | set(rng.choice(s2['entity_id'].to_numpy(), size=min(len(s2), a.n * 3), replace=False)))].reset_index(drop=True)
s3s = s3[s3['entity_id'].isin(need | set(rng.choice(s3['entity_id'].to_numpy(), size=min(len(s3), a.n * 3), replace=False)))].reset_index(drop=True)
for d in (s1s, s2s, s3s):
    d['_norm_name'] = d['business_name'].map(normalize_name); d['_norm_addr'] = d['business_address'].map(normalize_address)
t0 = time.time()
c2 = union_candidates(generate_candidates_token(s1s, s2s, max_df=a.max_df), generate_candidates_prefix(s1s, s2s))
c3 = union_candidates(generate_candidates_token(s1s, s3s, max_df=a.max_df), generate_candidates_prefix(s1s, s3s))
c = union_candidates(c2, c3)
r = candidate_recall(gts, c)
print(f"n={a.n} max_df={a.max_df} pairs={sum(len(v) for v in c.values()):,} recall={r:.4f} time={time.time()-t0:.1f}s", flush=True)
