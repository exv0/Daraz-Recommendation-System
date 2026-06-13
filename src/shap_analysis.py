"""
SHAP Feature Importance Analysis — Daraz Nepal Recommendation System
Trains a purchase-prediction classifier and explains it with SHAP.
Saves: data/shap_results.json
Author: Binnol Dahal | Coventry ID: 14809734
"""

import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import shap
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = DATA_DIR

print("=" * 55)
print("  SHAP Feature Importance Analysis")
print("=" * 55)

# ── 1. Load data ───────────────────────────────────────────────
print("\n📂 Loading data …")
interactions = pd.read_csv(f'{DATA_DIR}/cleaned_interactions.csv',
                           parse_dates=['timestamp'])
products     = pd.read_csv(f'{DATA_DIR}/product_features.csv')
users        = pd.read_csv(f'{DATA_DIR}/user_features.csv')

interactions['festival_context'] = (
    interactions['festival_context'].fillna('None').astype(str)
)
interactions.loc[interactions['festival_context'] == 'nan',
                 'festival_context'] = 'None'

# ── 2. Build purchase prediction dataset ───────────────────────
print("\n🔧 Building feature matrix …")

# Label: 1 = purchase, 0 = view/wishlist
interactions['label'] = (interactions['interaction_type'] == 'purchase').astype(int)

# Merge user + product features onto interactions
user_cols    = ['user_id', 'age', 'income_enc', 'device_enc',
                'remittance_enc', 'gender_enc', 'RFM_score', 'frequency']
product_cols = ['product_id', 'price_npr', 'rating', 'review_count',
                'category_enc', 'price_tier_enc', 'festival_enc', 'popularity_count']

user_cols    = [c for c in user_cols    if c in users.columns]
product_cols = [c for c in product_cols if c in products.columns]

df = (interactions[['user_id', 'product_id', 'label', 'is_festival',
                     'month', 'hour', 'dayofweek']]
      .merge(users[user_cols],    on='user_id',    how='left')
      .merge(products[product_cols], on='product_id', how='left'))

df = df.fillna(0)

# Feature columns for SHAP
feature_cols = [
    'age', 'income_enc', 'device_enc', 'remittance_enc',
    'gender_enc', 'RFM_score', 'frequency',
    'price_npr', 'rating', 'review_count',
    'category_enc', 'price_tier_enc', 'festival_enc', 'popularity_count',
    'is_festival', 'month', 'hour', 'dayofweek',
]
feature_cols = [c for c in feature_cols if c in df.columns]

# Friendly display names for the dashboard
FEATURE_LABELS = {
    'age':              'User Age',
    'income_enc':       'Income Level',
    'device_enc':       'Device Type',
    'remittance_enc':   'Remittance Receiver',
    'gender_enc':       'Gender',
    'RFM_score':        'RFM Score',
    'frequency':        'Purchase Frequency',
    'price_npr':        'Product Price (NPR)',
    'rating':           'Product Rating',
    'review_count':     'Review Count',
    'category_enc':     'Product Category',
    'price_tier_enc':   'Price Tier',
    'festival_enc':     'Festival Relevance',
    'popularity_count': 'Product Popularity',
    'is_festival':      'Festival Period',
    'month':            'Month of Year',
    'hour':             'Hour of Day',
    'dayofweek':        'Day of Week',
}

X = df[feature_cols].values
y = df['label'].values

# Sample for speed (keep class balance)
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]
n_sample = min(10_000, len(pos_idx))
neg_idx  = np.random.choice(neg_idx, n_sample, replace=False)
pos_idx  = np.random.choice(pos_idx, n_sample, replace=False)
idx      = np.concatenate([pos_idx, neg_idx])
np.random.shuffle(idx)
X_s, y_s = X[idx], y[idx]

print(f"   Training samples : {len(X_s):,} (balanced)")
print(f"   Features         : {len(feature_cols)}")

# ── 3. Train GradientBoosting classifier ──────────────────────
print("\n🤖 Training purchase-prediction classifier …")
X_train, X_test, y_train, y_test = train_test_split(
    X_s, y_s, test_size=0.2, random_state=42, stratify=y_s
)

clf = GradientBoostingClassifier(
    n_estimators=200, max_depth=4,
    learning_rate=0.1, random_state=42
)
clf.fit(X_train, y_train)

acc  = clf.score(X_test, y_test)
print(f"   Classifier accuracy : {acc:.3f}")

# ── 4. SHAP explanation ────────────────────────────────────────
print("\n🔍 Computing SHAP values …")
explainer   = shap.TreeExplainer(clf)
X_explain   = X_s[:500]                        # explain on 500 samples
shap_values = explainer.shap_values(X_explain)

# For binary classification, shap_values can be a list [neg_class, pos_class]
if isinstance(shap_values, list):
    sv = shap_values[1]      # positive class (purchase)
else:
    sv = shap_values

mean_abs_shap = np.abs(sv).mean(axis=0)

# Sort by importance
order        = np.argsort(mean_abs_shap)[::-1]
top_features = [(feature_cols[i],
                 FEATURE_LABELS.get(feature_cols[i], feature_cols[i]),
                 float(round(mean_abs_shap[i], 6)))
                for i in order]

print("\n  Top 10 features by SHAP importance:")
print(f"  {'Feature':<25}  {'SHAP value':>10}")
print("  " + "-" * 38)
for raw, label, val in top_features[:10]:
    print(f"  {label:<25}  {val:>10.4f}")

# ── 5. Save results ────────────────────────────────────────────
results = {
    'feature_names':   [f[0] for f in top_features],
    'feature_labels':  [f[1] for f in top_features],
    'shap_values':     [f[2] for f in top_features],
    'classifier_accuracy': float(round(acc, 4)),
    'n_samples':       int(len(X_s)),
    'n_features':      int(len(feature_cols)),
    'sklearn_importance': {
        feature_cols[i]: float(round(v, 6))
        for i, v in enumerate(clf.feature_importances_)
    }
}

with open(f'{OUTPUT_DIR}/shap_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ SHAP results saved → data/shap_results.json")
