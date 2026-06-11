"""
Content-Based Filtering Model
Daraz Nepal Recommendation System
Author: Binnol Dahal | Coventry ID: 14809734
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class ContentBasedFilter:
    """
    Content-Based Filtering using product feature vectors.
    - Builds a feature vector for each product (category, brand, price, rating …)
    - Builds a user profile by aggregating their interaction history
    - Recommends products most similar to the user's profile
    """

    def __init__(self):
        self.is_fitted  = False
        self.ohe        = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler     = MinMaxScaler()

    # ── Training ───────────────────────────────────────────────────────────────
    def fit(self, products_df, interactions_df):
        """
        Build product feature matrix and user profiles.
        products_df     : product_features.csv
        interactions_df : cleaned_interactions.csv
        """
        self.products_df     = products_df.set_index('product_id')
        self.product_ids     = list(self.products_df.index)
        self.iid_map         = {p: i for i, p in enumerate(self.product_ids)}
        self.interactions_df = interactions_df

        # ── Product feature matrix ─────────────────────────────────────────
        # Categorical: one-hot encode
        cat_cols = ['category', 'subcategory', 'brand',
                    'festival_relevance', 'price_tier']
        cat_data = products_df[cat_cols].fillna('Unknown')
        cat_enc  = self.ohe.fit_transform(cat_data)

        # Numerical: min-max scale
        num_cols = ['price_npr', 'rating', 'review_count']
        num_data = products_df[num_cols].fillna(0)
        num_enc  = self.scaler.fit_transform(num_data)

        # Combine into one feature matrix
        self.feature_matrix = np.hstack([cat_enc, num_enc]).astype(np.float32)
        print(f"   ✅ Product feature matrix: {self.feature_matrix.shape} "
              f"({cat_enc.shape[1]} categorical + {num_enc.shape[1]} numerical)")

        # ── User profiles from interaction history ─────────────────────────
        weight_map = {'view': 1, 'wishlist': 2, 'purchase': 5}
        interactions_df = interactions_df.copy()
        interactions_df['w'] = interactions_df['interaction_type'].map(weight_map)

        self.user_profiles = {}
        for uid, group in interactions_df.groupby('user_id'):
            vectors, weights = [], []
            for _, row in group.iterrows():
                pid = row['product_id']
                if pid in self.iid_map:
                    vectors.append(self.feature_matrix[self.iid_map[pid]])
                    weights.append(row['w'])
            if vectors:
                arr = np.vstack(vectors)
                w   = np.array(weights, dtype=np.float32).reshape(-1, 1)
                self.user_profiles[uid] = (arr * w).sum(axis=0) / w.sum()

        # Popularity fallback
        pop = interactions_df.groupby('product_id')['user_id'].nunique()
        self.popularity = {pid: pop.get(pid, 0) for pid in self.product_ids}

        self.is_fitted = True
        print(f"   ✅ User profiles built: {len(self.user_profiles):,}")
        return self

    # ── Inference ──────────────────────────────────────────────────────────────
    def recommend(self, user_id, n=10, exclude_seen=True):
        """Return top-N (product_id, score) based on content similarity."""
        if user_id not in self.user_profiles:
            # Cold-start: return top-N by popularity
            top = sorted(self.popularity, key=self.popularity.get, reverse=True)[:n]
            return [(pid, float(self.popularity[pid])) for pid in top]

        profile = self.user_profiles[user_id].reshape(1, -1)
        sims    = cosine_similarity(profile, self.feature_matrix)[0]

        if exclude_seen:
            seen = set(
                self.interactions_df[self.interactions_df['user_id'] == user_id]['product_id']
            )
            for pid in seen:
                if pid in self.iid_map:
                    sims[self.iid_map[pid]] = -1

        top_idx = np.argsort(sims)[::-1][:n]
        return [(self.product_ids[i], float(sims[i])) for i in top_idx if sims[i] > -1]

    def get_all_scores(self, user_id):
        """Return raw similarity scores for ALL items (used by hybrid model)."""
        if user_id not in self.user_profiles:
            return {pid: float(self.popularity.get(pid, 0)) for pid in self.product_ids}
        profile = self.user_profiles[user_id].reshape(1, -1)
        sims    = cosine_similarity(profile, self.feature_matrix)[0]
        return dict(zip(self.product_ids, sims.tolist()))

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path):
        joblib.dump(self, path)
        print(f"   Model saved → {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)
