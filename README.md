# Machine Learning-Based Product Recommendation System
### For Young Adult Customers on Daraz Nepal

**Author:** Binnol Dahal | **Coventry ID:** 14809734  
**Module:** ST6000CEM Individual Project Preparation  
**Module Leader:** Manoj Shrestha  

---

## Project Overview

A hybrid machine learning recommendation system tailored for young adult users (18–30) on Daraz Nepal. The system combines **Collaborative Filtering (SVD)**, **Content-Based Filtering**, and **Neural Collaborative Filtering (NCF)** into an adaptive hybrid ensemble, enriched with Nepal-specific signals such as festival periods (Dashain, Tihar), remittance-driven spending, and demographic segmentation.

---

## Features

- **4 ML Models:** Collaborative (SVD), Content-Based, Neural CF (NCF), Hybrid Ensemble
- **RFM Segmentation:** Users classified as Champions, Loyal, Potential, At Risk, Lost
- **Festival Context:** Dashain and Tihar seasonal boosts built into interaction data
- **SHAP Explainability:** Feature importance scores for every recommendation
- **Hyperparameter Tuning:** GridSearch (n_factors) + Bayesian optimisation (CF/CB weight)
- **Cold-Start Handling:** Popularity, demographic, and content-based fallback strategies
- **Ethical Safeguards:** Data privacy, filter bubble prevention, bias mitigation, informed consent
- **Live Dashboard:** 5-page React web app connected to Flask REST API

---

## Project Structure

```
Daraz Recommender/
├── data/                        # Raw and processed datasets
│   ├── users.csv
│   ├── products.csv
│   ├── interactions.csv
│   ├── cleaned_interactions.csv
│   ├── user_features.csv
│   ├── product_features.csv
│   ├── metrics.json             # Model evaluation results
│   ├── shap_results.json        # SHAP feature importance
│   └── tuning_results.json      # Hyperparameter tuning results
├── src/                         # Python backend
│   ├── generate_data.py         # Synthetic data generator
│   ├── preprocessing.py         # Feature engineering + RFM
│   ├── evaluate.py              # Model training + evaluation
│   ├── shap_analysis.py         # SHAP explainability
│   ├── hyperparameter_tuning.py # GridSearch + Bayesian tuning
│   ├── api/
│   │   └── app.py               # Flask REST API (12 endpoints)
│   └── models/
│       ├── collab_model.py      # SVD Collaborative Filtering
│       ├── content_model.py     # Content-Based Filtering
│       ├── hybrid_model.py      # Hybrid Ensemble
│       └── ncf_model.py         # Neural Collaborative Filtering
├── models_saved/                # Trained model files (.pkl)
├── frontend/                    # React dashboard
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   └── pages/
│   │       ├── Dashboard.jsx
│   │       ├── Recommendations.jsx
│   │       ├── Products.jsx
│   │       ├── ModelMetrics.jsx
│   │       └── EthicsAndColdStart.jsx
│   └── package.json
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data
```bash
cd src
python generate_data.py
```

### 3. Run preprocessing
```bash
python preprocessing.py
```

### 4. Train models and evaluate
```bash
python evaluate.py
```

### 5. Run SHAP analysis
```bash
python shap_analysis.py
```

### 6. Run hyperparameter tuning
```bash
python hyperparameter_tuning.py
```

### 7. Start Flask API
```bash
cd api
python app.py
```
API runs on `http://localhost:5000`

### 8. Start React dashboard (new terminal)
```bash
cd frontend
npm install
npm run dev
```
Dashboard runs on `http://localhost:3000`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/categories` | Category breakdown |
| GET | `/api/users` | Paginated user list |
| GET | `/api/user/<id>` | User profile + history |
| GET | `/api/products` | Product catalog |
| GET | `/api/product/<id>` | Single product detail |
| GET | `/api/recommend/<id>?model=hybrid` | Recommendations |
| GET | `/api/similar/<id>` | Similar products |
| GET | `/api/metrics` | Model evaluation metrics |
| GET | `/api/shap` | SHAP feature importance |
| POST | `/api/feedback` | User feedback |

---

## Dataset

| File | Records | Description |
|------|---------|-------------|
| `users.csv` | 10,000 | Young adults 18–30, Nepal cities |
| `products.csv` | 1,000 | 6 categories, Nepali brands, NPR pricing |
| `interactions.csv` | 446,028 | Views, wishlists, purchases with timestamps |

All data is **synthetic** and generated with Nepal-specific behavioural patterns including festival spikes, remittance-driven spending, and collectivist purchasing behaviour.

---

## Model Performance (K=10)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|-------|-------------|-----------|---------|
| Collaborative (SVD) | — | — | — |
| Content-Based | — | — | — |
| Neural CF (NCF) | — | — | — |
| **Hybrid (best)** | — | — | **highest** |

*Run `evaluate.py` to populate the metrics table above.*

---

## References

1. Ricci, F., Rokach, L., & Shapira, B. (2022). *Recommender Systems Handbook* (3rd ed.). Springer.
2. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques. *Computer*, 42(8).
3. Zhang, S. et al. (2019). Deep learning based recommender system. *ACM Computing Surveys*, 52(1).
4. Gomez-Uribe, C. A., & Hunt, N. (2015). The Netflix recommender system. *ACM TMIS*, 6(4).
5. Nepal Telecommunications Authority. (2025). *MIS Report Q4 2025*. Government of Nepal.