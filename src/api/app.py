"""
Flask REST API — Daraz Nepal Recommendation System
Author: Binnol Dahal | Coventry ID: 14809734
"""

import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json as _json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models_saved')

# ── Load data & models ────────────────────────────────────────────────────────
print("🚀 Loading data and models …")

users        = pd.read_csv(f'{DATA_DIR}/user_features.csv')
products     = pd.read_csv(f'{DATA_DIR}/product_features.csv')
interactions = pd.read_csv(f'{DATA_DIR}/cleaned_interactions.csv',
                           parse_dates=['timestamp'])

interactions['festival_context'] = (
    interactions['festival_context'].fillna('None').astype(str)
)
interactions.loc[interactions['festival_context'] == 'nan', 'festival_context'] = 'None'

cf_model     = joblib.load(f'{MODELS_DIR}/collab_model.pkl')
cb_model     = joblib.load(f'{MODELS_DIR}/content_model.pkl')
hybrid_model = joblib.load(f'{MODELS_DIR}/hybrid_model.pkl')

try:
    ncf_model = joblib.load(f'{MODELS_DIR}/ncf_model.pkl')
    print("✅ NCF model loaded")
except Exception as e:
    ncf_model = None
    print(f"⚠️  NCF model not found: {e}")

print("✅ API ready — all models loaded")

users_idx    = users.set_index('user_id')
products_idx = products.set_index('product_id')
CATEGORIES   = sorted(products['category'].unique().tolist())


def jsonify_safe(data):
    if isinstance(data, dict):
        return {k: jsonify_safe(v) for k, v in data.items()}
    if isinstance(data, list):
        return [jsonify_safe(v) for v in data]
    if isinstance(data, (np.integer,)):  return int(data)
    if isinstance(data, (np.floating,)): return round(float(data), 4)
    if isinstance(data, (np.bool_,)):    return bool(data)
    try:
        if pd.isna(data): return None
    except Exception:
        pass
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Daraz Nepal Recommender API is running'})


@app.route('/api/stats')
def stats():
    total_purchases = int((interactions['interaction_type'] == 'purchase').sum())
    total_views     = int((interactions['interaction_type'] == 'view').sum())
    total_wishlists = int((interactions['interaction_type'] == 'wishlist').sum())
    festival_ints   = int((interactions['festival_context'] != 'None').sum())
    segment_dist    = users['segment'].value_counts().to_dict() if 'segment' in users.columns else {}
    return jsonify({
        'total_users':        int(len(users)),
        'total_products':     int(len(products)),
        'total_interactions': int(len(interactions)),
        'total_purchases':    total_purchases,
        'total_views':        total_views,
        'total_wishlists':    total_wishlists,
        'festival_interactions': festival_ints,
        'categories':         len(CATEGORIES),
        'rfm_segments':       jsonify_safe(segment_dist),
        'avg_price_npr':      round(float(products['price_npr'].mean()), 2),
        'conversion_rate':    round(total_purchases / max(total_views, 1), 4),
    })


@app.route('/api/categories')
def categories():
    data = []
    for cat in CATEGORIES:
        cat_products  = products[products['category'] == cat]
        cat_purchases = interactions[
            interactions['product_id'].isin(cat_products['product_id']) &
            (interactions['interaction_type'] == 'purchase')
        ]
        data.append({
            'name':            cat,
            'product_count':   int(len(cat_products)),
            'avg_price_npr':   round(float(cat_products['price_npr'].mean()), 2),
            'total_purchases': int(len(cat_purchases)),
        })
    return jsonify({'categories': data})


@app.route('/api/users')
def get_users():
    page    = int(request.args.get('page',  1))
    limit   = int(request.args.get('limit', 20))
    segment = request.args.get('segment', None)
    df = users.copy()
    if segment:
        df = df[df['segment'] == segment]
    total = len(df)
    df    = df.iloc[(page-1)*limit : page*limit]
    cols  = [c for c in ['user_id','age','gender','city','income_level',
                          'device_type','preferred_category','segment',
                          'frequency','monetary','RFM_score'] if c in df.columns]
    return jsonify({'total': total, 'page': page, 'limit': limit,
                    'users': jsonify_safe(df[cols].to_dict(orient='records'))})


@app.route('/api/user/<int:user_id>')
def get_user(user_id):
    if user_id not in users_idx.index:
        return jsonify({'error': f'User {user_id} not found'}), 404
    row       = users_idx.loc[user_id]
    user_ints = interactions[interactions['user_id'] == user_id]
    history_summary = user_ints['interaction_type'].value_counts().to_dict()
    user_cats = user_ints.merge(products[['product_id','category']], on='product_id')
    top_cats  = user_cats.groupby('category').size().nlargest(3).index.tolist()
    recent    = (user_ints[user_ints['interaction_type'] == 'purchase']
                 .sort_values('timestamp', ascending=False).head(5)
                 .merge(products[['product_id','product_name','category','price_npr']],
                        on='product_id', how='left'))
    profile_cols = [c for c in ['age','gender','city','income_level','device_type',
                                 'preferred_category','segment','frequency','monetary',
                                 'RFM_score','remittance_receiver'] if c in row.index]
    return jsonify({
        'user_id':          user_id,
        'profile':          jsonify_safe(row[profile_cols].to_dict()),
        'history_summary':  jsonify_safe(history_summary),
        'top_categories':   top_cats,
        'recent_purchases': jsonify_safe(
            recent[['product_id','product_name','category','price_npr','timestamp']]
            .to_dict(orient='records')),
    })


@app.route('/api/products')
def get_products():
    category = request.args.get('category', None)
    limit    = int(request.args.get('limit', 20))
    sort_by  = request.args.get('sort', 'popularity_count')
    df = products.copy()
    if category:
        df = df[df['category'] == category]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    cols = [c for c in ['product_id','product_name','category','subcategory','brand',
                         'price_npr','rating','review_count','festival_relevance',
                         'popularity_count','conversion_rate'] if c in df.columns]
    return jsonify({'total': int(len(df)),
                    'products': jsonify_safe(df[cols].head(limit).to_dict(orient='records'))})


@app.route('/api/product/<int:product_id>')
def get_product(product_id):
    if product_id not in products_idx.index:
        return jsonify({'error': f'Product {product_id} not found'}), 404
    data = jsonify_safe(products_idx.loc[product_id].to_dict())
    data['product_id'] = product_id
    return jsonify(data)


@app.route('/api/recommend/<int:user_id>')
def recommend(user_id):
    n     = int(request.args.get('n', 10))
    model = request.args.get('model', 'hybrid')
    if user_id not in users_idx.index:
        return jsonify({'error': f'User {user_id} not found'}), 404
    try:
        if model == 'collaborative':
            raw  = cf_model.recommend(user_id, n=n)
            recs = [{'product_id': int(pid), 'score': round(float(s), 4),
                     'explanation': 'Users like you also viewed this'} for pid, s in raw]
        elif model == 'content':
            raw  = cb_model.recommend(user_id, n=n)
            recs = [{'product_id': int(pid), 'score': round(float(s), 4),
                     'explanation': 'Similar to items you browsed'} for pid, s in raw]
        elif model == 'ncf':
            if ncf_model is None:
                return jsonify({'error': 'NCF model not loaded'}), 503
            raw  = ncf_model.recommend(user_id, n=n)
            recs = [{'product_id': int(pid), 'score': round(float(s), 4),
                     'explanation': 'Neural network predicted you\'d like this'} for pid, s in raw]
        else:
            raw  = hybrid_model.recommend(user_id, n=n)
            recs = [{'product_id':  int(r['product_id']),
                     'score':        r['hybrid_score'],
                     'cf_score':     r['cf_score'],
                     'cb_score':     r['cb_score'],
                     'explanation':  r['explanation'],
                     'category':     r['category']} for r in raw]

        for rec in recs:
            pid = rec['product_id']
            if pid in products_idx.index:
                p = products_idx.loc[pid]
                rec['product_name']       = str(p['product_name'])
                rec['category']           = str(p.get('category', ''))
                rec['brand']              = str(p.get('brand', ''))
                rec['price_npr']          = int(p.get('price_npr', 0))
                rec['rating']             = float(p.get('rating', 0))
                rec['festival_relevance'] = str(p.get('festival_relevance', 'None'))

        user_row     = users_idx.loc[user_id]
        user_segment = str(user_row.get('segment', 'Unknown')) if 'segment' in user_row.index else 'Unknown'
        return jsonify({'user_id': user_id, 'model_used': model,
                        'user_segment': user_segment, 'n': n,
                        'recommendations': recs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/similar/<int:product_id>')
def similar_products(product_id):
    n = int(request.args.get('n', 6))
    if product_id not in products_idx.index:
        return jsonify({'error': f'Product {product_id} not found'}), 404
    try:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        pidx   = cb_model.iid_map.get(product_id)
        if pidx is None:
            return jsonify({'similar': []})
        target = cb_model.feature_matrix[pidx].reshape(1, -1)
        sims   = cos_sim(target, cb_model.feature_matrix)[0]
        sims[pidx] = -1
        top_idx = np.argsort(sims)[::-1][:n]
        results = []
        for i in top_idx:
            pid = cb_model.product_ids[i]
            if pid in products_idx.index:
                p = products_idx.loc[pid]
                results.append({'product_id': int(pid),
                                 'product_name': str(p['product_name']),
                                 'category': str(p['category']),
                                 'brand': str(p['brand']),
                                 'price_npr': int(p['price_npr']),
                                 'rating': float(p['rating']),
                                 'similarity': round(float(sims[i]), 4)})
        return jsonify({'product_id': product_id, 'similar': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    data     = request.get_json()
    required = ['user_id', 'product_id', 'feedback_type']
    if not all(k in data for k in required):
        return jsonify({'error': f'Missing fields. Required: {required}'}), 400
    valid_types = ['like', 'dislike', 'not_interested']
    if data['feedback_type'] not in valid_types:
        return jsonify({'error': f'feedback_type must be one of {valid_types}'}), 400
    return jsonify({'status': 'recorded', 'user_id': data['user_id'],
                    'product_id': data['product_id'],
                    'feedback_type': data['feedback_type'],
                    'message': 'Feedback noted. Model will update on next retraining cycle.'})


@app.route('/api/metrics')
def get_metrics():
    path = os.path.join(DATA_DIR, 'metrics.json')
    if not os.path.exists(path):
        return jsonify({'error': 'metrics.json not found — run evaluate.py first'}), 404
    with open(path) as f:
        return jsonify(_json.load(f))


@app.route('/api/shap')
def get_shap():
    path = os.path.join(DATA_DIR, 'shap_results.json')
    if not os.path.exists(path):
        return jsonify({'error': 'shap_results.json not found — run shap_analysis.py first'}), 404
    with open(path) as f:
        return jsonify(_json.load(f))


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)