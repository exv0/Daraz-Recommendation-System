"""
Collaborative Filtering Model — SVD Matrix Factorization
Daraz Nepal Recommendation System
Author: Binnol Dahal | Coventry ID: 14809734
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import joblib


class CollaborativeFilter:
    """
    SVD-based Collaborative Filtering.
    - Factorizes the user-item interaction matrix into latent factors
    - Falls back to popularity-based recommendations for cold-start users
    """

    def __init__(self, n_factors=50, random_state=42):
        self.n_factors    = n_factors
        self.svd          = TruncatedSVD(n_components=n_factors, random_state=random_state)
        self.is_fitted    = False

    # ── Training ───────────────────────────────────────────────────────────────
    def fit(self, user_item_df):
        """
        Fit SVD on a user-item DataFrame.
        Rows = users, Columns = product_ids, Values = interaction weights.
        """
        self.user_ids  = list(user_item_df.index)
        self.item_ids  = list(user_item_df.columns)
        self.uid_map   = {u: i for i, u in enumerate(self.user_ids)}
        self.iid_map   = {p: i for i, p in enumerate(self.item_ids)}

        matrix = user_item_df.values.astype(np.float32)

        # Decompose: user_factors (n_users × k), item_factors (n_items × k)
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_.T
        self.matrix       = matrix

        # Popularity scores for cold-start fallback
        self.popularity = (matrix > 0).sum(axis=0)

        self.is_fitted = True
        var_explained  = self.svd.explained_variance_ratio_.sum()
        print(f"   ✅ SVD fitted — {self.n_factors} factors, "
              f"variance explained: {var_explained:.1%}")
        return self

    # ── Inference ──────────────────────────────────────────────────────────────
    def _scores_for_user(self, user_idx):
        return self.user_factors[user_idx] @ self.item_factors.T

    def recommend(self, user_id, n=10, exclude_seen=True):
        """
        Return top-N (product_id, score) for a user.
        New users without history receive popularity-based fallback.
        """
        if user_id not in self.uid_map:
            # Cold-start fallback — most popular products
            top = np.argsort(self.popularity)[::-1][:n]
            return [(self.item_ids[i], float(self.popularity[i])) for i in top]

        uidx   = self.uid_map[user_id]
        scores = self._scores_for_user(uidx).copy()

        if exclude_seen:
            scores[self.matrix[uidx] > 0] = -np.inf

        top_idx = np.argsort(scores)[::-1][:n]
        return [
            (self.item_ids[i], float(scores[i]))
            for i in top_idx if scores[i] > -np.inf
        ]

    def get_all_scores(self, user_id):
        """Return raw scores for ALL items (used by hybrid model)."""
        if user_id not in self.uid_map:
            return dict(zip(self.item_ids, self.popularity.astype(float)))
        uidx   = self.uid_map[user_id]
        scores = self._scores_for_user(uidx)
        return dict(zip(self.item_ids, scores.tolist()))

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path):
        joblib.dump(self, path)
        print(f"   Model saved → {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)
