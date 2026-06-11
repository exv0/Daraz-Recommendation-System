"""
Synthetic Data Generator — Daraz Nepal Recommendation System
Generates: users.csv, products.csv, interactions.csv
Author: Binnol Dahal | Coventry ID: 14809734
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
N_USERS        = 10000
N_PRODUCTS     = 1000
N_INTERACTIONS = 60000
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Nepal-specific constants ───────────────────────────────────────────────────
CITIES = ['Kathmandu', 'Lalitpur', 'Bhaktapur', 'Pokhara',
          'Biratnagar', 'Butwal', 'Birgunj', 'Dharan', 'Chitwan', 'Hetauda']
CITY_WEIGHTS = [0.33, 0.14, 0.09, 0.12, 0.08, 0.06, 0.05, 0.05, 0.05, 0.03]

# Category definitions: price range (NPR), festival boosters, catalog weight
CATEGORIES = {
    'Electronics': {
        'subcategories': ['Smartphones', 'Laptops', 'Headphones', 'Cameras',
                          'Smart Watches', 'Tablets', 'Speakers'],
        'brands':        ['Samsung', 'Apple', 'Xiaomi', 'Realme',
                          'OnePlus', 'HP', 'Dell', 'Asus', 'Vivo', 'Oppo'],
        'price_range':   (5_000, 150_000),
        'festival':      {'Dashain': 2.8, 'Tihar': 1.6, 'None': 1.0},
        'weight':        0.25,
    },
    'Fashion': {
        'subcategories': ['T-Shirts', 'Jeans', 'Kurta', 'Saree',
                          'Jackets', 'Shoes', 'Ethnic Wear', 'Kurtis'],
        'brands':        ["Wrangler", "Levi's", "Allen Solly",
                          "Nepali Designs", "Cottonworld", "Zara"],
        'price_range':   (300, 15_000),
        'festival':      {'Dashain': 1.6, 'Tihar': 3.2, 'None': 1.0},
        'weight':        0.20,
    },
    'Phone Accessories': {
        'subcategories': ['Cases', 'Screen Protectors', 'Chargers',
                          'Power Banks', 'Earbuds', 'Cables', 'Stands'],
        'brands':        ['Baseus', 'Anker', 'Ugreen', 'Belkin', 'Generic'],
        'price_range':   (150, 6_000),
        'festival':      {'Dashain': 1.4, 'Tihar': 1.3, 'None': 1.0},
        'weight':        0.20,
    },
    'Beauty & Personal Care': {
        'subcategories': ['Skincare', 'Haircare', 'Makeup',
                          'Perfume', "Men's Grooming", 'Sunscreen'],
        'brands':        ['Loreal', 'Garnier', 'Nivea',
                          'Himalaya', 'Biotique', "Dove", 'Mamaearth'],
        'price_range':   (150, 8_000),
        'festival':      {'Dashain': 1.9, 'Tihar': 2.2, 'None': 1.0},
        'weight':        0.15,
    },
    'Home & Kitchen': {
        'subcategories': ['Cookware', 'Storage', 'Bedding',
                          'Decor', 'Cleaning', 'Air Purifiers'],
        'brands':        ['Prestige', 'Pigeon', 'Milton', 'Tefal', 'Local'],
        'price_range':   (200, 25_000),
        'festival':      {'Dashain': 2.1, 'Tihar': 1.9, 'None': 1.0},
        'weight':        0.10,
    },
    'Sports & Fitness': {
        'subcategories': ['Gym Equipment', 'Yoga', 'Cricket',
                          'Football', 'Badminton', 'Outdoor Gear'],
        'brands':        ['Nivia', 'Cosco', 'Vector X', 'Yonex', 'Generic'],
        'price_range':   (400, 30_000),
        'festival':      {'Dashain': 1.3, 'Tihar': 1.1, 'None': 1.0},
        'weight':        0.10,
    },
}

# Festival windows (approx dates in 2024 for realistic timestamps)
FESTIVAL_WINDOWS = {
    'Dashain': (datetime(2024, 10, 2), datetime(2024, 10, 15)),
    'Tihar':   (datetime(2024, 10, 29), datetime(2024, 11, 4)),
}

# ── Helper functions ───────────────────────────────────────────────────────────

def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def get_festival(dt: datetime) -> str:
    for name, (start, end) in FESTIVAL_WINDOWS.items():
        if start <= dt <= end:
            return name
    return 'None'


# ── 1. Generate Users ──────────────────────────────────────────────────────────
print("⏳ Generating users …")

signup_start = datetime(2022, 1, 1)
signup_end   = datetime(2024, 6, 1)

users = []
for uid in range(1, N_USERS + 1):
    age    = int(np.random.normal(24, 3.5))
    age    = max(18, min(30, age))          # clamp to 18-30
    gender = np.random.choice(['Male', 'Female', 'Other'],
                               p=[0.54, 0.44, 0.02])
    city   = np.random.choice(CITIES, p=CITY_WEIGHTS)
    income = np.random.choice(['Low', 'Medium', 'High'],
                               p=[0.40, 0.45, 0.15])
    device = np.random.choice(['Mobile', 'Desktop', 'Tablet'],
                               p=[0.76, 0.19, 0.05])
    remittance = np.random.choice([True, False], p=[0.28, 0.72])  # 28% receiver
    signup_dt  = random_date(signup_start, signup_end)

    # Preferred categories vary by demographic
    cat_weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
    if gender == 'Female':
        cat_weights = [0.15, 0.30, 0.15, 0.25, 0.10, 0.05]
    if age < 22:
        cat_weights = [0.30, 0.20, 0.25, 0.10, 0.05, 0.10]

    preferred_cat = np.random.choice(list(CATEGORIES.keys()), p=cat_weights)

    users.append({
        'user_id':         uid,
        'age':             age,
        'gender':          gender,
        'city':            city,
        'income_level':    income,
        'device_type':     device,
        'remittance_receiver': remittance,
        'preferred_category':  preferred_cat,
        'signup_date':     signup_dt.strftime('%Y-%m-%d'),
    })

df_users = pd.DataFrame(users)
df_users.to_csv(f'{OUTPUT_DIR}/users.csv', index=False)
print(f"   ✅ {len(df_users)} users saved")


# ── 2. Generate Products ───────────────────────────────────────────────────────
print("⏳ Generating products …")

cat_names   = list(CATEGORIES.keys())
cat_weights = [CATEGORIES[c]['weight'] for c in cat_names]

products = []
for pid in range(1, N_PRODUCTS + 1):
    cat  = np.random.choice(cat_names, p=cat_weights)
    info = CATEGORIES[cat]

    subcat = random.choice(info['subcategories'])
    brand  = random.choice(info['brands'])
    lo, hi = info['price_range']

    # Price: log-normal so we get realistic right-skewed distribution
    price = int(np.random.lognormal(
        mean=np.log((lo + hi) / 2), sigma=0.6
    ))
    price = max(lo, min(hi, price))

    rating       = round(np.random.beta(8, 2) * 4 + 1, 1)   # 1–5, skewed high
    review_count = int(np.random.lognormal(4, 1.2))          # right-skewed

    # Assign a primary festival relevance label
    fest_label = max(info['festival'],
                     key=lambda k: info['festival'][k] if k != 'None' else 0)
    if info['festival'][fest_label] <= 1.0:
        fest_label = 'None'

    products.append({
        'product_id':         pid,
        'product_name':       f"{brand} {subcat} #{pid}",
        'category':           cat,
        'subcategory':        subcat,
        'brand':              brand,
        'price_npr':          price,
        'rating':             rating,
        'review_count':       review_count,
        'festival_relevance': fest_label,
        'in_stock':           np.random.choice([True, False], p=[0.90, 0.10]),
    })

df_products = pd.DataFrame(products)
df_products.to_csv(f'{OUTPUT_DIR}/products.csv', index=False)
print(f"   ✅ {len(df_products)} products saved")


# ── 3. Generate Interactions ───────────────────────────────────────────────────
print("⏳ Generating interactions …")

interaction_start = datetime(2024, 1, 1)
interaction_end   = datetime(2024, 12, 31)

# Pre-index products by category for fast lookup
cat_to_pids = {cat: df_products[df_products['category'] == cat]['product_id'].tolist()
               for cat in cat_names}

interactions = []
iid = 1

for row in df_users.itertuples():
    uid = row.user_id

    # How many interactions? Heavier users are more active
    n_sessions = int(np.random.lognormal(3.5, 0.8))   # ~30 median sessions
    n_sessions = max(5, min(300, n_sessions))

    for _ in range(n_sessions):
        ts = random_date(interaction_start, interaction_end)
        festival = get_festival(ts)

        # Pick category: prefer user's preferred cat, boosted during festivals
        weights = []
        for cat in cat_names:
            w = 1.0
            if cat == row.preferred_category:
                w *= 2.5
            if festival != 'None':
                w *= CATEGORIES[cat]['festival'].get(festival, 1.0)
            weights.append(w)
        total = sum(weights)
        weights = [w / total for w in weights]
        chosen_cat = np.random.choice(cat_names, p=weights)

        # Pick a product from that category
        pids_in_cat = cat_to_pids.get(chosen_cat, df_products['product_id'].tolist())
        pid = random.choice(pids_in_cat)

        product_row = df_products[df_products['product_id'] == pid].iloc[0]
        price = product_row['price_npr']

        # Determine affordability based on income
        afford_prob = {'Low': 0.4, 'Medium': 0.7, 'High': 0.9}[row.income_level]
        if price > 50_000:
            afford_prob *= 0.3
        elif price > 20_000:
            afford_prob *= 0.6

        # Interaction funnel: view → wishlist → purchase
        interaction_type = 'view'
        r = random.random()
        if r < afford_prob * 0.12:
            interaction_type = 'purchase'
        elif r < afford_prob * 0.30:
            interaction_type = 'wishlist'

        # Rating: only for purchases
        rating = None
        if interaction_type == 'purchase':
            rating = int(np.random.choice([1, 2, 3, 4, 5],
                                           p=[0.05, 0.07, 0.13, 0.35, 0.40]))

        interactions.append({
            'interaction_id':   iid,
            'user_id':          uid,
            'product_id':       pid,
            'interaction_type': interaction_type,
            'timestamp':        ts.strftime('%Y-%m-%d %H:%M:%S'),
            'festival_context': festival,
            'rating':           rating,
            'session_id':       iid * 7 % (N_USERS * 10),   # pseudo session id
        })
        iid += 1

df_interactions = pd.DataFrame(interactions)
df_interactions.to_csv(f'{OUTPUT_DIR}/interactions.csv', index=False)
print(f"   ✅ {len(df_interactions)} interactions saved")


# ── 4. Quick sanity summary ────────────────────────────────────────────────────
print("\n📊 Dataset Summary")
print("=" * 45)
print(f"  Users        : {len(df_users):,}")
print(f"  Products     : {len(df_products):,}")
print(f"  Interactions : {len(df_interactions):,}")
print(f"\n  Interaction breakdown:")
print(df_interactions['interaction_type'].value_counts().to_string())
print(f"\n  Festival distribution:")
print(df_interactions['festival_context'].value_counts().to_string())
print(f"\n  Product category spread:")
print(df_products['category'].value_counts().to_string())
print(f"\n  User city spread:")
print(df_users['city'].value_counts().to_string())
print("\n✅ All files saved to /data/")
