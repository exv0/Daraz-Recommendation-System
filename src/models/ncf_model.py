"""
Neural Collaborative Filtering (NCF)
Daraz Nepal Recommendation System
Uses SVD latent factors as embeddings fed into a Multi-Layer Perceptron.
Author: Binnol Dahal | Coventry ID: 14809734
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib


class NeuralCollaborativeFilter:
    """
    NCF: learns non-linear user-item interaction patterns via MLP.

    Architecture:
      [user SVD factors (50d)] ──┐
                                  ├──► concat (100d) ──► MLP ──► score
      [item SVD factors (50d)] ──┘

    Builds on top of the CollaborativeFilter's SVD embeddings — the SVD
    captures linear structure; the MLP captures non-linear patterns on top.
    """

    def __init__(self, hidden_layers=(128, 64, 32), random_state=42):
        self.mlp = MLPRegressor(
            hidden_layer_sizes = hidden_layers,
            activation         = 'relu',
            solver             = 'adam',
            max_iter           = 300,
            random_state       = random_state,
            early_stopping     = True,
            validation_fraction= 0.1,
            n_iter_no_change   = 15,
            verbose            = False,
        )
        self.scaler    = StandardScaler()
        self.is_fitted = False

    # ── Training ───────────────────────────────────────────────────────────────
    def fit(self, user_item_df, cf_model, max_pos=25_000):
        """
        Train NCF.
        user_item_df : user-item interaction matrix (from evaluate.py)
        cf_model     : fitted CollaborativeFilter (supplies SVD embeddings)
        max_pos      : cap positive training samples for speed
        """
        self.user_ids = list(user_item_df.index)
        self.item_ids = list(user_item_df.columns)
        self.uid_map  = {u: i for i, u in enumerate(self.user_ids)}
        self.iid_map  = {p: i for i, p in enumerate(self.item_ids)}

        # SVD embeddings from pre-trained CF model
        self.user_factors = cf_model.user_factors   # (n_users, k)
        self.item_factors = cf_model.item_factors   # (n_items, k)
        self.matrix       = user_item_df.values.astype(np.float32)
        self.popularity   = (self.matrix > 0).sum(axis=0)

        # ── Build training set ─────────────────────────────────────────────
        rows, cols = np.where(self.matrix > 0)

        # Cap positives
        if len(rows) > max_pos:
            idx  = np.random.choice(len(rows), max_pos, replace=False)
            rows = rows[idx];  cols = cols[idx]

        X_pos = np.hstack([self.user_factors[rows], self.item_factors[cols]])
        y_pos = self.matrix[rows, cols]

        # Random negative sampling (same count as positives)
        n_neg = len(rows)
        X_neg, neg_added = [], 0
        attempts = 0
        while neg_added < n_neg and attempts < n_neg * 10:
            r = np.random.randint(len(self.user_ids))
            c = np.random.randint(len(self.item_ids))
            if self.matrix[r, c] == 0:
                X_neg.append(np.concatenate([self.user_factors[r],
                                              self.item_factors[c]]))
                neg_added += 1
            attempts += 1

        X_neg = np.array(X_neg)
        y_neg = np.zeros(len(X_neg))

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([y_pos, y_neg])

        # Shuffle
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]

        # Scale and train
        X_scaled = self.scaler.fit_transform(X)
        self.mlp.fit(X_scaled, y)

        self.is_fitted = True
        print(f"   ✅ NCF fitted — {len(X):,} training samples, "
              f"loss: {self.mlp.loss_:.4f}, "
              f"iterations: {self.mlp.n_iter_}")
        return self

    # ── Inference ──────────────────────────────────────────────────────────────
    def _user_vec(self, user_id):
        if user_id in self.uid_map:
            return self.user_factors[self.uid_map[user_id]]
        return self.user_factors.mean(axis=0)   # cold-start fallback

    def recommend(self, user_id, n=10, exclude_seen=True):
        """Return top-N (product_id, score) tuples."""
        if user_id not in self.uid_map:
            top = np.argsort(self.popularity)[::-1][:n]
            return [(self.item_ids[i], float(self.popularity[i])) for i in top]

        u_vec = self._user_vec(user_id)
        X_all = np.hstack([
            np.tile(u_vec, (len(self.item_ids), 1)),
            self.item_factors
        ])
        scores = self.mlp.predict(self.scaler.transform(X_all))

        if exclude_seen:
            uidx = self.uid_map[user_id]
            scores[self.matrix[uidx] > 0] = -np.inf

        top_idx = np.argsort(scores)[::-1][:n]
        return [
            (self.item_ids[i], float(scores[i]))
            for i in top_idx if scores[i] > -np.inf
        ]

    def get_all_scores(self, user_id):
        """Return scores for all items — used by hybrid model."""
        if user_id not in self.uid_map:
            return {self.item_ids[i]: float(self.popularity[i])
                    for i in range(len(self.item_ids))}

        u_vec = self._user_vec(user_id)
        X_all = np.hstack([
            np.tile(u_vec, (len(self.item_ids), 1)),
            self.item_factors
        ])
        scores = self.mlp.predict(self.scaler.transform(X_all))
        return dict(zip(self.item_ids, scores.tolist()))

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path):
        joblib.dump(self, path)
        print(f"   NCF model saved → {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)