"""CPU train-only: blocking -> features -> LightGBM OOF -> threshold. No test inference."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'business_entity_resolution', 'src'))
from io_utils import load_source, load_ground_truth
from normalize import normalize_name, normalize_address
from blocking import generate_candidates_token, generate_candidates_prefix, union_candidates, candidate_recall
from pipeline import build_pair_frame, subsample_negatives
from features import build_feature_frame_vectorized, FEATURE_NAMES
from model import train_oof, calibrate_oof, tune_threshold
from metric import macro_f05
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', default='dataset')
ap.add_argument('--n', type=int, default=5000)
ap.add_argument('--max-df', type=int, default=300)
ap.add_argument('--n-splits', type=int, default=3)
ap.add_argument('--seed', type=int, default=42)
a = ap.parse_args()
t0=time.time()
s1=load_source(os.path.join(a.data_dir,'train','train_source1.tsv'),'S1-')
s2=load_source(os.path.join(a.data_dir,'train','train_source2.tsv'),'S2-')
s3=load_source(os.path.join(a.data_dir,'train','train_source3.tsv'),'S3-')
gt=load_ground_truth(os.path.join(a.data_dir,'train','train_ground_truth.tsv'))
print(f"load {time.time()-t0:.1f}s", flush=True)
rng=np.random.default_rng(a.seed)
keep=set(rng.choice(s1['entity_id'].to_numpy(), size=min(a.n,len(s1)), replace=False))
s1s=s1[s1['entity_id'].isin(keep)].reset_index(drop=True)
gts={k:v for k,v in gt.items() if k in keep}
need=set().union(*gts.values()) if gts else set()
s2s=s2[s2['entity_id'].isin(need|set(rng.choice(s2['entity_id'].to_numpy(), size=min(len(s2),a.n*3), replace=False)))].reset_index(drop=True)
s3s=s3[s3['entity_id'].isin(need|set(rng.choice(s3['entity_id'].to_numpy(), size=min(len(s3),a.n*3), replace=False)))].reset_index(drop=True)
for d in (s1s,s2s,s3s):
    d['_norm_name']=d['business_name'].map(normalize_name); d['_norm_addr']=d['business_address'].map(normalize_address)
c2=union_candidates(generate_candidates_token(s1s,s2s,max_df=a.max_df), generate_candidates_prefix(s1s,s2s))
c3=union_candidates(generate_candidates_token(s1s,s3s,max_df=a.max_df), generate_candidates_prefix(s1s,s3s))
import pandas as pd
pairs=pd.concat([build_pair_frame(s1s,s2s,c2), build_pair_frame(s1s,s3s,c3)], ignore_index=True)
feat=build_feature_frame_vectorized(pairs)
labels=np.array([1 if o in gts.get(s,set()) else 0 for s,o in zip(feat['s1_id'],feat['other_id'])])
print(f"pairs={len(feat):,} pos={labels.sum():,} recall={candidate_recall(gts,union_candidates(c2,c3)):.4f}", flush=True)
feat_s, lab_s = subsample_negatives(feat, labels, 15, seed=a.seed)
print(f"train pairs={len(feat_s):,} pos_rate={lab_s.mean():.4f}", flush=True)
oof, models = train_oof(feat_s, lab_s, n_splits=a.n_splits, seed=a.seed)
iso = calibrate_oof(oof, lab_s)
cal = iso.predict(oof)
bt, bs = tune_threshold(feat_s, cal, gts, s1s['entity_id'].tolist())
base = macro_f05(gts, {e:set() for e in s1s['entity_id']}, entity_ids=s1s['entity_id'].tolist())
print(f"OOF macro F0.5={bs:.4f} @t={bt:.3f} | singleton-baseline={base:.4f}", flush=True)
