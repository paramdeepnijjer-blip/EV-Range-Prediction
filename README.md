# 🚗 EV Range Prediction – Data Science Mini Project

Predicting the **electric driving range** of vehicles registered in **Washington State** using data science and machine learning.  
This project demonstrates a complete end-to-end ML workflow — from data cleaning to feature engineering, model comparison, and interpretability.

---

## 📊 Overview

This project analyzes a real-world dataset of **Electric Vehicle (EV)** registrations and predicts the vehicle’s **electric range (in miles)** based on key features:

- `model_year` – Year of the vehicle model  
- `make` – Manufacturer (top 20 brands included)  
- `base_msrp` – Manufacturer’s Suggested Retail Price  
- `electric_vehicle_type` – Battery Electric (BEV) vs Plug-in Hybrid (PHEV)  
- `eligible` – Clean Alternative Fuel Vehicle (CAFV) eligibility  

The analysis also explores **trends in EV technology and adoption** through visualizations and partial dependence plots.

---

<details>
<summary>🎯 <b>Objectives</b></summary>

- Clean and preprocess raw vehicle registration data  
- Develop regression models (Linear Regression vs Random Forest)  
- Evaluate model performance using RMSE and R² metrics  
- Interpret model results using feature importance and PDPs  
- Visualize trends in EV adoption and range improvements  
</details>

---

<details>
<summary>🧠 <b>Methodology</b></summary>

1. **Data Cleaning**
   - Dropped missing `model_year` and `make` values  
   - Standardized text fields (`city`, `county`, `make`)  
   - Converted numeric columns to appropriate datatypes  

2. **Feature Engineering**
   - Used `OneHotEncoder` for categorical variables  
   - Applied `SimpleImputer` (median/mode) for missing values  
   - Scaled numeric features using `StandardScaler`  

3. **Modeling**
   - Built two regression models with scikit-learn Pipelines:
     - **Linear Regression** (baseline)
     - **Random Forest Regressor** (improved)
</details>

---

## ⚙️ Model Performance

| Model | RMSE | R² |
|:------|:-----:|:--:|
| Linear Regression | ≈ 24.7 | 0.91 |
| Random Forest (100 trees) | **≈ 7.05** | **0.99** |

✅ **Random Forest** performed significantly better, capturing non-linear relationships between vehicle features and range.

---

<details>
<summary>🔍 <b>Key Findings</b></summary>

- **Model year** is the strongest predictor — newer models have longer ranges.  
- **Vehicle type (BEV vs PHEV)** plays a major role in determining range.  
- **Base MSRP** shows only weak correlation once type/year are considered.  
- **Feature importance** confirms technological advancement drives range gains.
</details>

---

<details>
<summary>📈 <b>Visualization Insights</b></summary>

- **Distribution of Electric Range:** Right-skewed — most vehicles have <50 miles range, a few reach 300+.  
- **Electric Range vs Model Year:** Strong upward trend after 2015.  
- **Electric Range vs MSRP:** No clear trend — expensive doesn’t always mean higher range.  
- **Residual Plots:** Centered around zero — confirms well-fitted model.  
- **PDPs:** Reinforce `model_year` and `vehicle_type` as key positive predictors.  
</details>

---

<details>
<summary>⚠️ <b>Limitations</b></summary>

- `model_year` used as a proxy for registration year — not exact sale date.  
- Dataset covers only **Washington State**, may not generalize nationally.  
- High-cardinality features (`make`, `model`) were limited to top-20 values.  
- Median/mode imputation slightly smooths data variability.  
- Random Forest is a black-box model — future work could use SHAP or LIME.  
</details>

---

<details>
<summary>🧩 <b>Tools & Libraries</b></summary>

- **Python 3.10+**  
- **Pandas**, **NumPy**, **Matplotlib**, **Scikit-Learn**  
- **Jupyter Notebook** for analysis and documentation  
</details>

---

## 🧾 Results Summary

> “The Random Forest model achieved near-perfect accuracy (R² ≈ 0.99), confirming that **technological progress and powertrain type** are the main drivers of EV range improvements.”

---

## 💡 Future Work

- Add multi-state or national-level datasets for generalization  
- Introduce time-series forecasting for future EV range trends  
- Include infrastructure and charging station variables  
- Use SHAP-based explainability for deeper model insights  

---

## 🧑‍💻 Author

**Paramdeep Nijjer**  
📍 Edmonton, AB — Data Science & Applied Statistics  
🔗 [GitHub](https://github.com/paramdeepnijjer-blip) | [LinkedIn](https://www.linkedin.com/)  

---

⭐ *If you found this project interesting, consider giving it a star!*
