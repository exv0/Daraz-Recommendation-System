"""
Model Training & Evaluation — Daraz Nepal Recommendation System
Metrics: Precision@K, Recall@K, F1@K, NDCG@K
Author: Binnol Dahal | Coventry ID: 14809734
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from models.collab_model   import CollaborativeFilter
from models.content_model  import ContentBasedFilter
from models.hybrid_model   import HybridRecommender

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models_saved')
os.makedirs(MODELS_DIR, exist_ok=True)

K          = 10          # evaluate Precision/Recall/NDCG at K=10
EVAL_USERS = 400         # sample size for evaluation (speed vs accuracy trade-off)


# ── Metric functions ───────────────────────────────────────────────────────────
def precision_at_k(recommended, relevant, k):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k

def recall_at_k(recommended, relevant, k):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / max(len(relevant), 1)

def f1_at_k(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def ndcg_at_k(recommended, relevant, k):
    dcg  = sum(1 / np.log2(i + 2)
               for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / max(idcg, 1e-10)

def evaluate_model(recommend_fn, test_user_items, label, k=K):
    """
    Evaluate a recommendation function on held-out test interactions.
    recommend_fn : callable(user_id) → list of product_ids
    test_user_items : dict { user_id → [relevant product_ids] }
    """
    metrics = {'precision': [], 'recall': [], 'f1': [], 'ndcg': []}

    for uid, relevant in test_user_items.items():
        try:
            recs = recommend_fn(uid)
        except Exception:
            continue

        p = precision_at_k(recs, relevant, k)
        r = recall_at_k(recs, relevant, k)
        metrics['precision'].append(p)
        metrics['recall'].append(r)
        metrics['f1'].append(f1_at_k(p, r))
        metrics['ndcg'].append(ndcg_at_k(recs, relevant, k))

    result = {m: np.mean(v) for m, v in metrics.items()}
    print(f"   {label:<22}  "
          f"P@{k}={result['precision']:.4f}  "
          f"R@{k}={result['recall']:.4f}  "
          f"F1={result['f1']:.4f}  "
          f"NDCG={result['ndcg']:.4f}")
    return result


# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 3 — Model Training & Evaluation")
print("=" * 60)

# ── 1. Load Data ───────────────────────────────────────────────
print("\n📂 Loading data …")
interactions = pd.read_csv(f'{DATA_DIR}/cleaned_interactions.csv',
                           parse_dates=['timestamp'])
products     = pd.read_csv(f'{DATA_DIR}/product_features.csv')
users        = pd.read_csv(f'{DATA_DIR}/user_features.csv')

# Fix festival_context
interactions['festival_context'] = interactions['festival_context'].fillna('None').astype(str)
interactions.loc[interactions['festival_context'] == 'nan', 'festival_context'] = 'None'

print(f"   Interactions : {len(interactions):,}")
print(f"   Products     : {len(products):,}")
print(f"   Users        : {len(users):,}")


# ── 2. Time-based Train / Test Split ──────────────────────────
print("\n✂️  Splitting data (time-based 80/20) …")
interactions.sort_values('timestamp', inplace=True)
split_idx   = int(len(interactions) * 0.8)
split_time  = interactions.iloc[split_idx]['timestamp']

train_df = interactions[interactions['timestamp'] <  split_time].copy()
test_df  = interactions[interactions['timestamp'] >= split_time].copy()

print(f"   Train : {len(train_df):,} interactions  (before {split_time.date()})")
print(f"   Test  : {len(test_df):,}  interactions  (after  {split_time.date()})")

# Build test ground-truth: purchases only (most meaningful signal)
test_purchases = test_df[test_df['interaction_type'] == 'purchase']
test_user_items = (
    test_purchases.groupby('user_id')['product_id']
    .apply(list).to_dict()
)

# Keep only users with ≥ 2 test purchases and who appear in train
train_users      = set(train_df['user_id'].unique())
eval_candidates  = {
    uid: pids for uid, pids in test_user_items.items()
    if len(pids) >= 2 and uid in train_users
}
np.random.seed(42)
eval_user_ids   = np.random.choice(list(eval_candidates.keys()),
                                    size=min(EVAL_USERS, len(eval_candidates)),
                                    replace=False)
eval_user_items = {uid: eval_candidates[uid] for uid in eval_user_ids}
print(f"   Eval users   : {len(eval_user_items)}")


# ── 3. Build User-Item Matrix (train set only) ─────────────────
print("\n🔢 Building training user-item matrix …")
weight_map  = {'view': 1, 'wishlist': 2, 'purchase': 5}
train_df['w'] = train_df['interaction_type'].map(weight_map)

matrix_data = train_df.groupby(['user_id', 'product_id'])['w'].max().reset_index()
user_item   = matrix_data.pivot(index='user_id', columns='product_id', values='w').fillna(0)
print(f"   Matrix shape  : {user_item.shape}")


# ── 4. Train Models ────────────────────────────────────────────
print("\n🤖 Training models …")

# Collaborative Filter
print("\n  [1/3] Collaborative Filtering (SVD)")
cf = CollaborativeFilter(n_factors=50)
cf.fit(user_item)
cf.save(f'{MODELS_DIR}/collab_model.pkl')

# Content-Based Filter
print("\n  [2/3] Content-Based Filtering")
cb = ContentBasedFilter()
cb.fit(products, train_df)
cb.save(f'{MODELS_DIR}/content_model.pkl')

# Hybrid
print("\n  [3/3] Hybrid Ensemble")
hybrid = HybridRecommender(cf, cb, train_df, products)
hybrid.save(f'{MODELS_DIR}/hybrid_model.pkl')
print("   ✅ Hybrid model ready")


# ── 5. Evaluate ────────────────────────────────────────────────
print(f"\n📊 Evaluating models (K={K}, n_users={len(eval_user_items)}) …\n")
print(f"   {'Model':<22}  {'P@K':>6}  {'R@K':>6}  {'F1':>6}  {'NDCG':>6}")
print("   " + "-" * 52)

def cf_rec(uid):
    return [pid for pid, _ in cf.recommend(uid, n=K)]

def cb_rec(uid):
    return [pid for pid, _ in cb.recommend(uid, n=K)]

def hybrid_rec(uid):
    return [r['product_id'] for r in hybrid.recommend(uid, n=K)]

results = {}
results['Collaborative (SVD)']  = evaluate_model(cf_rec,     eval_user_items, 'Collaborative (SVD)')
results['Content-Based']        = evaluate_model(cb_rec,     eval_user_items, 'Content-Based')
results['Hybrid (CF + CB)']     = evaluate_model(hybrid_rec, eval_user_items, 'Hybrid (CF + CB)')

# ── 6. Results Table ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  📈 Final Model Comparison")
print("=" * 60)
df_results = pd.DataFrame(results).T.round(4)
df_results.index.name = 'Model'
print(df_results.to_string())

# Identify best model
best_model = max(results, key=lambda m: results[m]['ndcg'])
print(f"\n  🏆 Best model by NDCG@{K}: {best_model}")


# ── 7. Sample Recommendations ──────────────────────────────────
print("\n" + "=" * 60)
print("  🎯 Sample Recommendations (Hybrid Model)")
print("=" * 60)

sample_users = list(eval_user_items.keys())[:3]
for uid in sample_users:
    recs = hybrid.recommend(uid, n=5)
    print(f"\n  User {uid} | "
          f"Purchases: {len(eval_user_items[uid])} test items")
    print(f"  {'Product ID':>10}  {'Category':<25}  {'Score':>6}  Explanation")
    print("  " + "-" * 65)
    for r in recs:
        print(f"  {r['product_id']:>10}  {r['category']:<25}  "
              f"{r['hybrid_score']:>6.4f}  {r['explanation']}")

print("\n✅ All models trained and evaluated successfully!")
print(f"   Saved to: {MODELS_DIR}/")

# ── 8. Train NCF & evaluate ─────────────────────────────────────────────────
import json
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))
from ncf_model import NeuralCollaborativeFilter

print("\n  [4/4] Neural Collaborative Filtering (NCF)")
ncf = NeuralCollaborativeFilter(hidden_layers=(128, 64, 32))
ncf.fit(user_item, cf)
ncf.save(f'{MODELS_DIR}/ncf_model.pkl')

def ncf_rec(uid):
    return [pid for pid, _ in ncf.recommend(uid, n=K)]

results['Neural CF (NCF)'] = evaluate_model(ncf_rec, eval_user_items, 'Neural CF (NCF)')

# ── 9. Save metrics to JSON ─────────────────────────────────────────────────
metrics_out = {
    'k':           K,
    'n_eval_users': len(eval_user_items),
    'models':      {k: {m: round(v, 6) for m, v in r.items()}
                    for k, r in results.items()},
    'best_model':  max(results, key=lambda m: results[m]['ndcg']),
    'generated_at': datetime.now().isoformat(),
}
metrics_path = os.path.join(DATA_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_out, f, indent=2)

print(f"\n✅ Metrics saved → {metrics_path}")