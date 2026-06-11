"""
Hybrid Ensemble Model — Collaborative + Content-Based + Diversity
Daraz Nepal Recommendation System
Author: Binnol Dahal | Coventry ID: 14809734
"""

import numpy as np
import pandas as pd
import joblib


class HybridRecommender:
    """
    Adaptive hybrid model combining Collaborative Filtering and Content-Based Filtering.

    Adaptive weighting:
      - New users (< 5 interactions) → 20% CF / 80% CB  (cold-start safe)
      - Regular users                → 60% CF / 40% CB
      - Power users (> 50 purchases) → 80% CF / 20% CB

    Post-processing:
      - Diversity filter: ensures recommendations span at least 3 categories
      - Explainability: returns a human-readable reason tag for each recommendation
    """

    CF_WEIGHT_DEFAULT  = 0.60
    CB_WEIGHT_DEFAULT  = 0.40
    CF_WEIGHT_COLD     = 0.20   # cold-start users
    CB_WEIGHT_COLD     = 0.80
    CF_WEIGHT_POWER    = 0.80   # power users
    CB_WEIGHT_POWER    = 0.20

    def __init__(self, cf_model, cb_model, interactions_df, products_df):
        self.cf          = cf_model
        self.cb          = cb_model
        self.interactions = interactions_df
        self.products     = products_df.set_index('product_id')
        self.is_fitted    = True

        # Pre-compute interaction counts per user (for adaptive weighting)
        counts = interactions_df.groupby('user_id')['interaction_id'].count()
        self.interaction_counts = counts.to_dict()

        purchases = interactions_df[interactions_df['interaction_type'] == 'purchase']
        pcounts   = purchases.groupby('user_id')['product_id'].count()
        self.purchase_counts = pcounts.to_dict()

    # ── Adaptive weight selection ───────────────────────────────────────────────
    def _get_weights(self, user_id):
        n_int  = self.interaction_counts.get(user_id, 0)
        n_buy  = self.purchase_counts.get(user_id, 0)
        if n_int < 5:
            return self.CF_WEIGHT_COLD, self.CB_WEIGHT_COLD
        if n_buy > 50:
            return self.CF_WEIGHT_POWER, self.CB_WEIGHT_POWER
        return self.CF_WEIGHT_DEFAULT, self.CB_WEIGHT_DEFAULT

    # ── Score normalisation ────────────────────────────────────────────────────
    @staticmethod
    def _minmax(score_dict):
        vals = np.array(list(score_dict.values()), dtype=np.float32)
        lo, hi = vals.min(), vals.max()
        if hi == lo:
            return {k: 0.5 for k in score_dict}
        return {k: float((v - lo) / (hi - lo)) for k, v in score_dict.items()}

    # ── Explainability tag ────────────────────────────────────────────────────
    def _explain(self, product_id, user_id, cf_score, cb_score, cf_w, cb_w):
        """Return a short human-readable reason for the recommendation."""
        category = self.products.loc[product_id, 'category'] \
            if product_id in self.products.index else 'item'
        festival = self.products.loc[product_id, 'festival_relevance'] \
            if product_id in self.products.index else 'None'

        if festival != 'None':
            return f"Popular during {festival}"
        if cf_w > cb_w and cf_score > cb_score:
            return f"Users like you also bought this {category}"
        if cb_score > cf_score:
            return f"Similar to {category} you've browsed"
        return f"Trending in {category}"

    # ── Main recommendation method ────────────────────────────────────────────
    def recommend(self, user_id, n=10, diversity=True, exclude_seen=True):
        """
        Return top-N recommendations as list of dicts:
          { product_id, hybrid_score, cf_score, cb_score, category, explanation }
        """
        cf_w, cb_w = self._get_weights(user_id)

        # Raw scores from both models
        cf_scores_raw = self.cf.get_all_scores(user_id)
        cb_scores_raw = self.cb.get_all_scores(user_id)

        # Normalise to [0, 1] for fair combination
        cf_scores = self._minmax(cf_scores_raw)
        cb_scores = self._minmax(cb_scores_raw)

        # Hybrid score
        all_items = set(cf_scores) | set(cb_scores)
        hybrid    = {
            pid: cf_w * cf_scores.get(pid, 0) + cb_w * cb_scores.get(pid, 0)
            for pid in all_items
        }

        # Remove seen items
        if exclude_seen:
            seen = set(
                self.interactions[self.interactions['user_id'] == user_id]['product_id']
            )
            for pid in seen:
                hybrid.pop(pid, None)

        # Sort by hybrid score
        ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)

        # Diversity post-processing: ensure ≥ 3 categories in top-N
        if diversity and n >= 6:
            ranked = self._diversify(ranked, n)
        else:
            ranked = ranked[:n]

        # Build output with explanations
        results = []
        for pid, h_score in ranked:
            cat = self.products.loc[pid, 'category'] \
                  if pid in self.products.index else 'Unknown'
            results.append({
                'product_id':    pid,
                'hybrid_score':  round(h_score, 4),
                'cf_score':      round(cf_scores.get(pid, 0), 4),
                'cb_score':      round(cb_scores.get(pid, 0), 4),
                'category':      cat,
                'explanation':   self._explain(pid, user_id,
                                               cf_scores.get(pid, 0),
                                               cb_scores.get(pid, 0),
                                               cf_w, cb_w),
            })
        return results

    # ── Diversity filter (MMR-inspired) ───────────────────────────────────────
    def _diversify(self, ranked, n):
        """
        Greedy diversity: pick top item, then next item that either
        scores highly OR comes from a category not yet represented.
        """
        selected    = []
        seen_cats   = set()
        remaining   = list(ranked)

        while remaining and len(selected) < n:
            # Try to find a new category if we have < 3 so far
            if len(seen_cats) < 3:
                for item in remaining:
                    pid = item[0]
                    cat = self.products.loc[pid, 'category'] \
                          if pid in self.products.index else 'Unknown'
                    if cat not in seen_cats:
                        selected.append(item)
                        seen_cats.add(cat)
                        remaining.remove(item)
                        break
                else:
                    # All remaining are same category — just take top
                    selected.append(remaining.pop(0))
            else:
                selected.append(remaining.pop(0))

        return selected

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path):
        joblib.dump(self, path)
        print(f"   Hybrid model saved → {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)
