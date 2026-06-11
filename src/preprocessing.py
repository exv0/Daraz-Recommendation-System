"""
Preprocessing, EDA & Feature Engineering — Daraz Nepal Recommendation System
Outputs:
  - data/cleaned_interactions.csv
  - data/user_features.csv          (RFM + demographic features)
  - data/product_features.csv       (content features)
  - data/user_item_matrix.csv       (sparse interaction matrix)
Author: Binnol Dahal | Coventry ID: 14809734
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = DATA_DIR   # save processed files alongside raw

# ── 1. Load Raw Data ───────────────────────────────────────────────────────────
print("=" * 55)
print("  STEP 2 — Preprocessing & Feature Engineering")
print("=" * 55)

print("\n📂 Loading raw data …")
users        = pd.read_csv(f'{DATA_DIR}/users.csv')
products     = pd.read_csv(f'{DATA_DIR}/products.csv')
interactions = pd.read_csv(f'{DATA_DIR}/interactions.csv')

print(f"   Users        : {len(users):,}")
print(f"   Products     : {len(products):,}")
print(f"   Interactions : {len(interactions):,}")


# ── 2. Clean Interactions ──────────────────────────────────────────────────────
print("\n🧹 Cleaning interactions …")

# Parse timestamps
interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
interactions['date']      = interactions['timestamp'].dt.date
interactions['month']     = interactions['timestamp'].dt.month
interactions['hour']      = interactions['timestamp'].dt.hour
interactions['dayofweek'] = interactions['timestamp'].dt.dayofweek  # 0=Mon

# Fix festival_context: 'None' string may load as NaN
interactions['festival_context'] = interactions['festival_context'].fillna('None').astype(str)
interactions.loc[interactions['festival_context'] == 'nan', 'festival_context'] = 'None'

# Drop duplicates
before = len(interactions)
interactions.drop_duplicates(subset=['user_id', 'product_id', 'interaction_type', 'timestamp'],
                              inplace=True)
print(f"   Dropped {before - len(interactions):,} duplicate rows")

# Check for nulls
null_summary = interactions.isnull().sum()
print(f"   Null check (rating is expected null for non-purchases):")
print(f"     rating nulls = {null_summary['rating']:,} "
      f"(purchases = {(interactions.interaction_type=='purchase').sum():,}) ✅")

# Interaction type → numeric weight (used for implicit feedback)
TYPE_WEIGHT = {'view': 1, 'wishlist': 2, 'purchase': 5}
interactions['interaction_weight'] = interactions['interaction_type'].map(TYPE_WEIGHT)

# Is festival period?
interactions['is_festival'] = interactions['festival_context'] != 'None'

print(f"   ✅ Cleaned interactions: {len(interactions):,} rows")
interactions.to_csv(f'{OUTPUT_DIR}/cleaned_interactions.csv', index=False)


# ── 3. RFM Feature Engineering (User-Level) ───────────────────────────────────
print("\n📊 Engineering RFM features …")

REFERENCE_DATE = pd.Timestamp('2025-01-01')   # 1 day after dataset end

purchases = interactions[interactions['interaction_type'] == 'purchase'].copy()

rfm = purchases.groupby('user_id').agg(
    recency   = ('timestamp', lambda x: (REFERENCE_DATE - x.max()).days),
    frequency = ('product_id', 'count'),
    monetary  = ('product_id', lambda x: products.loc[
                    products['product_id'].isin(x), 'price_npr'].sum())
).reset_index()

# Users with zero purchases get max recency, zero freq/monetary
all_users = pd.DataFrame({'user_id': users['user_id']})
rfm = all_users.merge(rfm, on='user_id', how='left')
rfm['recency']   = rfm['recency'].fillna((REFERENCE_DATE - pd.Timestamp('2024-01-01')).days)
rfm['frequency'] = rfm['frequency'].fillna(0).astype(int)
rfm['monetary']  = rfm['monetary'].fillna(0)

# RFM scores 1–5 (5 = best)
def score_col(series, ascending=True):
    """Bin into quintiles 1-5."""
    labels = [1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]
    try:
        return pd.qcut(series, q=5, labels=labels, duplicates='drop').astype(float)
    except Exception:
        return pd.cut(series, bins=5, labels=labels).astype(float)

rfm['R_score'] = score_col(rfm['recency'],   ascending=False)  # lower recency = better
rfm['F_score'] = score_col(rfm['frequency'], ascending=True)
rfm['M_score'] = score_col(rfm['monetary'],  ascending=True)
rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# Customer segments
def segment(score):
    if score >= 12:  return 'Champions'
    if score >= 9:   return 'Loyal'
    if score >= 6:   return 'Potential'
    if score >= 3:   return 'At Risk'
    return 'Lost'

rfm['segment'] = rfm['RFM_score'].apply(segment)
print(f"   RFM segment distribution:")
print(rfm['segment'].value_counts().to_string(header=False))


# ── 4. User Behavioral Features ───────────────────────────────────────────────
print("\n🧠 Building user behavioral features …")

# Category affinity: which category does each user interact with most?
user_cat = interactions.merge(products[['product_id','category']], on='product_id')
user_cat_agg = user_cat.groupby(['user_id','category'])['interaction_weight'].sum().reset_index()
top_cat = user_cat_agg.sort_values('interaction_weight', ascending=False)\
                       .groupby('user_id').first().reset_index()\
                       .rename(columns={'category':'top_category'})

# Wishlist rate
wl = interactions.groupby('user_id').apply(
    lambda x: (x['interaction_type']=='wishlist').sum() / max(len(x),1)
).reset_index().rename(columns={0:'wishlist_rate'})

# Festival engagement flag
fest_users = interactions[interactions['is_festival']]['user_id'].unique()

# Session diversity: how many unique categories browsed
cat_diversity = user_cat.groupby('user_id')['category'].nunique().reset_index()\
                         .rename(columns={'category':'category_diversity'})

# Avg session hour (morning/afternoon/evening shopper)
avg_hour = interactions.groupby('user_id')['hour'].mean().reset_index()\
                        .rename(columns={'hour':'avg_shop_hour'})

# Build user features table
user_features = users.merge(rfm, on='user_id')\
                      .merge(top_cat[['user_id','top_category']], on='user_id', how='left')\
                      .merge(wl, on='user_id', how='left')\
                      .merge(cat_diversity, on='user_id', how='left')\
                      .merge(avg_hour, on='user_id', how='left')

user_features['is_festival_shopper'] = user_features['user_id'].isin(fest_users)

# Encode categoricals for ML
user_features['gender_enc']      = user_features['gender'].map({'Male':0,'Female':1,'Other':2})
user_features['income_enc']      = user_features['income_level'].map({'Low':0,'Medium':1,'High':2})
user_features['device_enc']      = user_features['device_type'].map({'Mobile':0,'Desktop':1,'Tablet':2})
user_features['remittance_enc']  = user_features['remittance_receiver'].astype(int)
user_features['festival_enc']    = user_features['is_festival_shopper'].astype(int)

user_features.to_csv(f'{OUTPUT_DIR}/user_features.csv', index=False)
print(f"   ✅ User features saved: {user_features.shape[1]} columns × {len(user_features):,} rows")


# ── 5. Product Content Features ───────────────────────────────────────────────
print("\n📦 Building product content features …")

# Price tiers (NPR)
def price_tier(p):
    if p < 1_000:   return 'Budget'
    if p < 5_000:   return 'Mid'
    if p < 20_000:  return 'Premium'
    return 'Luxury'

products['price_tier'] = products['price_npr'].apply(price_tier)

# Popularity: how many unique users interacted with this product
popularity = interactions.groupby('product_id')['user_id'].nunique().reset_index()\
                          .rename(columns={'user_id':'popularity_count'})

# Purchase rate per product
prod_stats = interactions.groupby('product_id')['interaction_type'].value_counts().unstack(fill_value=0)
prod_stats.columns.name = None
for col in ['view','wishlist','purchase']:
    if col not in prod_stats.columns:
        prod_stats[col] = 0
prod_stats['conversion_rate'] = prod_stats['purchase'] / (prod_stats['view'] + 1)
prod_stats = prod_stats.reset_index()

product_features = products.merge(popularity, on='product_id', how='left')\
                            .merge(prod_stats[['product_id','view','wishlist',
                                               'purchase','conversion_rate']],
                                   on='product_id', how='left')

product_features['popularity_count'] = product_features['popularity_count'].fillna(0).astype(int)
product_features['conversion_rate']  = product_features['conversion_rate'].fillna(0)

# Encode categoricals
cat_enc = {c:i for i,c in enumerate(products['category'].unique())}
product_features['category_enc'] = product_features['category'].map(cat_enc)

price_tier_enc = {'Budget':0,'Mid':1,'Premium':2,'Luxury':3}
product_features['price_tier_enc'] = product_features['price_tier'].map(price_tier_enc)

festival_enc_map = {'None':0,'Dashain':1,'Tihar':2}
product_features['festival_enc'] = product_features['festival_relevance'].map(festival_enc_map).fillna(0)

product_features.to_csv(f'{OUTPUT_DIR}/product_features.csv', index=False)
print(f"   ✅ Product features saved: {product_features.shape[1]} columns × {len(product_features):,} rows")


# ── 6. User-Item Interaction Matrix ───────────────────────────────────────────
print("\n🔢 Building user-item interaction matrix …")

# Pivot: rows = users, cols = products, values = max interaction weight
matrix_data = interactions.groupby(['user_id','product_id'])['interaction_weight'].max().reset_index()

user_item_matrix = matrix_data.pivot(index='user_id', columns='product_id', values='interaction_weight')
user_item_matrix = user_item_matrix.fillna(0)

# Sparsity check
total_cells  = user_item_matrix.shape[0] * user_item_matrix.shape[1]
filled_cells = (user_item_matrix > 0).sum().sum()
sparsity     = 1 - filled_cells / total_cells

print(f"   Matrix shape  : {user_item_matrix.shape[0]:,} users × {user_item_matrix.shape[1]:,} products")
print(f"   Filled cells  : {filled_cells:,} / {total_cells:,}")
print(f"   Sparsity      : {sparsity:.2%}  (typical RS matrices: >95% — this is realistic)")

user_item_matrix.to_csv(f'{OUTPUT_DIR}/user_item_matrix.csv')
print(f"   ✅ User-item matrix saved")


# ── 7. Final Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ✅ PREPROCESSING COMPLETE — Files saved to /data/")
print("=" * 55)
print(f"""
  📄 cleaned_interactions.csv  — {len(interactions):>7,} rows
  📄 user_features.csv         — {len(user_features):>7,} rows, {user_features.shape[1]} features
  📄 product_features.csv      — {len(product_features):>7,} rows, {product_features.shape[1]} features
  📄 user_item_matrix.csv      — {user_item_matrix.shape[0]:>4,} × {user_item_matrix.shape[1]} matrix

  Key stats:
    RFM Champions   : {(rfm['segment']=='Champions').sum():,} users
    Loyal           : {(rfm['segment']=='Loyal').sum():,} users
    Avg purchases/user : {rfm['frequency'].mean():.1f}
    Avg spend/user (NPR): {rfm['monetary'].mean():,.0f}
    Matrix sparsity : {sparsity:.2%}
""")
