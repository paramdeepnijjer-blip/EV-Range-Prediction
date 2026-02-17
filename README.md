# ⚡ Washington State EV Adoption Analysis & Range Prediction

**Mini Project | Exploratory Data Analysis + Machine Learning**

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![R²](https://img.shields.io/badge/R²-0.992-brightgreen?style=for-the-badge)

---

## 🎯 Project Overview

This project analyzes **264,628 registered electric vehicle records** across Washington State to uncover adoption trends, brand dominance, geographic distribution, and builds a machine learning model to predict electric range.

**Two-part analysis:**
1. **Exploratory Data Analysis** — trends, brands, geography
2. **Predictive Modeling** — predict electric range with ML

---

## 🔑 Key Results

| Model | RMSE | R² |
|-------|------|-----|
| Linear Regression (baseline) | 24.67 | 0.906 |
| **Random Forest (final)** | **7.05** | **0.992** |

- **Random Forest explains 99.2% of variance** in electric range
- **Top predictor: `model_year`** — newer vehicles travel farther
- EV registrations grew **exponentially** after 2020, peaking at **60,157 in 2023**
- **Tesla dominates** Washington's EV market by a wide margin
- **King County** leads geographic adoption

---

## 📊 Dataset

- **Source:** Washington State Department of Licensing (DOL)
- **Size:** 264,628 registered EV records, 21 features
- **Key features:**

| Feature | Description |
|---------|-------------|
| `model_year` | Year of vehicle model |
| `make` | Vehicle manufacturer |
| `electric_vehicle_type` | BEV (fully electric) or PHEV (plug-in hybrid) |
| `electric_range` | Miles per charge (target variable) |
| `base_msrp` | Manufacturer suggested retail price |
| `county` / `city` | Geographic location |
| `eligible` | CAFV clean fuel eligibility |

---

## 📈 EDA Findings

### 🚗 Adoption Trend
- EV registrations grew **exponentially from 2010 to 2023**
- **Peak year: 2023** with 60,157 new registrations
- Post-2020 surge driven by expanded model availability and state incentives

### 🏆 Top EV Brands
1. **Tesla** — dominant market leader
2. **Chevrolet** — strong second
3. **Nissan** — third place
4. Ford, BMW, Toyota, Kia, Volkswagen, Hyundai, Jeep round out top 10

### 📍 Geographic Distribution
- **King County** leads with highest registrations
- Followed by **Snohomish** and **Pierce** counties
- Urban density and charging infrastructure drive adoption

### 🔋 Electric Range Insights
- Most vehicles have range **below 50 miles** (PHEVs)
- High-end BEVs reach **300+ miles**
- Right-skewed distribution showing wide variability
- **Range increases steadily by model year** — clear technological progress

---

## 🤖 Machine Learning Pipeline

### Feature Engineering
```python
features = ['model_year', 'base_msrp', 'make', 
            'electric_vehicle_type', 'eligible']
target = 'electric_range'
```

### Preprocessing Pipeline
```python
Pipeline([
    ('pre', ColumnTransformer([
        ('num', Pipeline([imputer, scaler]), num_cols),
        ('cat', Pipeline([imputer, onehot_encoder]), cat_cols)
    ])),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])
```

### Model Results
- **Training size:** 16,000 samples (subsampled for speed)
- **Test size:** 4,000 samples (20%)
- Random Forest **dramatically outperforms** Linear Regression baseline

### Feature Importance (Top Predictors)
1. 🥇 **`model_year`** — dominant predictor (technological progress)
2. 🥈 **`electric_vehicle_type`** — BEVs have higher range than PHEVs
3. 🥉 **`make`** — manufacturer differences in battery technology
4. **`base_msrp`** — higher price = longer range
5. **`eligible`** — CAFV eligibility reflects modern EV specs

### Residual Diagnostics
- ✅ Residuals centered at zero (no bias)
- ✅ Random scatter around zero line (homoscedasticity)
- ✅ R² consistently **0.97–0.99 across all model year groups** (strong generalization)

### Partial Dependence Plots
- `model_year` shows positive slope until ~2020 (improving battery tech)
- Sharp post-2020 dip likely due to **incomplete registration data** for newest models
- `base_msrp` shows positive relationship with electric range

---

## 🗂️ Repository Structure

```
washington-ev-analysis/
├── Mini_Project.ipynb         # Full analysis notebook
├── EV_Cleaned.csv             # Cleaned dataset (264K records)
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## 🚀 Getting Started

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook Mini_Project.ipynb
```

### Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## ⚠️ Limitations

- `model_year` is a proxy — not actual purchase date
- Dataset limited to **Washington State** (may not generalize nationally)
- `make` and `model` capped to top 20 brands (rare brands excluded)
- Missing values imputed with median/mode
- Random Forest is a black-box — SHAP values would improve interpretability

---

## 🔮 Future Work

1. **Hyperparameter tuning** — GridSearchCV for `n_estimators`, `max_depth`
2. **Advanced models** — XGBoost, Gradient Boosting comparison
3. **SHAP values** — Explainable AI for feature impact per prediction
4. **Include `model` name** — finer granularity in predictions
5. **Dashboard** — Tableau or Streamlit for interactive exploration
6. **National comparison** — extend analysis beyond Washington State

---

## 💡 Key Takeaways

> **Technological progress (model year) and vehicle type (BEV vs PHEV) are the strongest predictors of electric range — not brand or price.**

> **Washington EV adoption is accelerating rapidly, concentrated in urban King County, with Tesla maintaining overwhelming market dominance.**

---

## 👤 Author

**Paramdeep Nijjer**
- GitHub: [@paramdeepnijjer-bliip](https://github.com/paramdeepnijjer-bliip)
- LinkedIn: [https://www.linkedin.com/in/paramdeepnijjer/]

---

## 📄 License

MIT License — feel free to use and build on this analysis.

---

*Analysis based on Washington State DOL EV registration data*
