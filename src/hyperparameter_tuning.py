"""
Hyperparameter Tuning — Daraz Nepal Recommendation System
Methods: GridSearchCV + Bayesian Optimisation (scipy minimize)
Saves: data/tuning_results.json
Author: Binnol Dahal | Coventry ID: 14809734
"""

import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import TruncatedSVD
from scipy.optimize import minimize

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

print("=" * 60)
print("  Hyperparameter Tuning — GridSearch + Bayesian")
print("=" * 60)

# ── Load & split data ──────────────────────────────────────────
print("\n📂 Loading data …")
interactions = pd.read_csv(f'{DATA_DIR}/cleaned_interactions.csv',
                           parse_dates=['timestamp'])
interactions['festival_context'] = (
    interactions['festival_context'].fillna('None').astype(str))
interactions = interactions.sort_values('timestamp')
weight_map   = {'view': 1, 'wishlist': 2, 'purchase': 5}
interactions['w'] = interactions['interaction_type'].map(weight_map)

# 70% train / 30% test split (within tuning)
split_idx  = int(len(interactions) * 0.70)
train_df   = interactions.iloc[:split_idx]
test_df    = interactions.iloc[split_idx:]

# Build matrices
def build_matrix(df):
    mg = df.groupby(['user_id','product_id'])['w'].max().reset_index()
    return mg.pivot(index='user_id', columns='product_id', values='w').fillna(0)

train_mat = build_matrix(train_df)
test_mat  = build_matrix(test_df)

# Common users and items
common_users = train_mat.index.intersection(test_mat.index)
common_items = train_mat.columns.intersection(test_mat.columns)

train_sub = train_mat.loc[common_users, common_items]
test_sub  = test_mat.loc[common_users, common_items]

M_train = train_sub.values.astype(np.float32)
M_test  = test_sub.values.astype(np.float32)
user_ids = list(train_sub.index)
item_ids = list(train_sub.columns)

print(f"   Train matrix : {M_train.shape[0]:,} × {M_train.shape[1]:,}")
print(f"   Test  matrix : {M_test.shape[0]:,}  × {M_test.shape[1]:,}")

# ── Evaluation helper ──────────────────────────────────────────
np.random.seed(42)
active_users = np.where((M_train.sum(axis=1) > 3) &
                         (M_test.sum(axis=1)  > 0))[0]
eval_sample  = np.random.choice(active_users,
                                 size=min(300, len(active_users)),
                                 replace=False)
print(f"   Eval users   : {len(eval_sample)}")

def evaluate_svd(n_factors):
    svd  = TruncatedSVD(n_components=int(n_factors), random_state=42)
    U    = svd.fit_transform(M_train)
    V    = svd.components_.T
    pred = U @ V.T

    precisions, recalls = [], []
    for uidx in eval_sample:
        scores = pred[uidx].copy()
        scores[M_train[uidx] > 0] = -np.inf     # exclude train items
        top10    = set(np.argsort(scores)[::-1][:10].tolist())
        relevant = set(np.where(M_test[uidx] > 0)[0].tolist())
        if not relevant:
            continue
        hits = len(top10 & relevant)
        precisions.append(hits / 10)
        recalls.append(hits / len(relevant))

    return float(np.mean(precisions)), float(np.mean(recalls))


# ── 1. GridSearch — n_factors ──────────────────────────────────
print("\n🔍 GridSearch: n_factors ∈ {10, 20, 30, 50, 75, 100} …")
print(f"   {'n_factors':>9}  {'P@10':>8}  {'R@10':>8}")
print("   " + "-" * 30)

grid_results = []
for n in [10, 20, 30, 50, 75, 100]:
    p, r = evaluate_svd(n)
    grid_results.append({'n_factors': n, 'precision_at_10': round(p, 6),
                          'recall_at_10': round(r, 6)})
    print(f"   {n:>9}  {p:>8.6f}  {r:>8.6f}")

best_grid = max(grid_results, key=lambda x: x['precision_at_10'])
print(f"\n   ✅ Best: n_factors={best_grid['n_factors']} "
      f"(P@10={best_grid['precision_at_10']:.6f})")


# ── 2. Bayesian Optimisation — hybrid CF/CB weight ─────────────
print("\n🧠 Bayesian Optimisation: hybrid CF weight ∈ [0.1, 0.9] …")

best_n = best_grid['n_factors']
svd    = TruncatedSVD(n_components=best_n, random_state=42)
U      = svd.fit_transform(M_train)
V      = svd.components_.T
cf_raw = U @ V.T

# CB proxy: global item popularity from training set
pop    = M_train.sum(axis=0)
cb_raw = np.tile(pop, (M_train.shape[0], 1))

# Normalise 0-1
def norm(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)

cf_n = norm(cf_raw)
cb_n = norm(cb_raw)

def neg_precision_hybrid(params):
    cf_w = float(np.clip(params[0], 0.05, 0.95))
    cb_w = 1 - cf_w
    precs = []
    for uidx in eval_sample:
        hybrid = cf_w * cf_n[uidx] + cb_w * cb_n[uidx]
        hybrid[M_train[uidx] > 0] = -np.inf
        top10    = set(np.argsort(hybrid)[::-1][:10].tolist())
        relevant = set(np.where(M_test[uidx] > 0)[0].tolist())
        if not relevant:
            continue
        precs.append(len(top10 & relevant) / 10)
    return -float(np.mean(precs)) if precs else 0.0

print(f"   {'cf_weight':>9}  {'cb_weight':>9}  {'P@10':>8}")
print("   " + "-" * 32)

bayesian_trials = []
for cf_w_init in [0.3, 0.5, 0.6, 0.7, 0.8]:
    result = minimize(neg_precision_hybrid, x0=[cf_w_init],
                      method='L-BFGS-B', bounds=[(0.05, 0.95)],
                      options={'maxiter': 30, 'ftol': 1e-8})
    cf_w = round(float(np.clip(result.x[0], 0.05, 0.95)), 4)
    cb_w = round(1 - cf_w, 4)
    p    = round(-float(result.fun), 6)
    bayesian_trials.append({'cf_weight': cf_w, 'cb_weight': cb_w,
                             'precision_at_10': p})
    print(f"   {cf_w:>9.3f}  {cb_w:>9.3f}  {p:>8.6f}")

best_bayes = max(bayesian_trials, key=lambda x: x['precision_at_10'])
print(f"\n   ✅ Best: CF={best_bayes['cf_weight']}, CB={best_bayes['cb_weight']} "
      f"(P@10={best_bayes['precision_at_10']:.6f})")


# ── 3. Save results ────────────────────────────────────────────
results = {
    'gridsearch': {
        'parameter': 'n_factors (SVD)',
        'values':    grid_results,
        'best':      best_grid,
    },
    'bayesian': {
        'parameter': 'hybrid CF/CB weight',
        'trials':    bayesian_trials,
        'best':      best_bayes,
    },
    'summary': {
        'optimal_n_factors': best_grid['n_factors'],
        'optimal_cf_weight': best_bayes['cf_weight'],
        'optimal_cb_weight': best_bayes['cb_weight'],
    }
}

with open(f'{DATA_DIR}/tuning_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved → data/tuning_results.json")