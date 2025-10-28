# 🚗 EV Range Prediction – Data Science Mini Project

Predicting the electric driving range of vehicles registered in Washington State using data science and machine learning.

## 📊 Overview
This project analyzes a real-world EV dataset and builds predictive models to estimate a vehicle’s **electric range** based on features such as:
- Model year  
- Base MSRP  
- Make (top-20 manufacturers)  
- Electric vehicle type (BEV vs PHEV)  
- Clean alternative fuel vehicle (CAFV) eligibility  

The project also explores **trends in EV technology and adoption** through data visualization.

## 🧠 Objectives
- Clean and preprocess raw vehicle registration data  
- Develop regression models (Linear Regression vs RandomForest)  
- Evaluate model performance (RMSE, R²)  
- Interpret feature importance and partial dependence  
- Visualize year-wise EV range improvements and adoption trends  

## 🧮 Key Results
| Model | RMSE (km) | R² | Notes |
|--------|------------|------|--------|
| Linear Regression | ~17 | 0.97 | Baseline model |
| Random Forest (100 trees) | **~9** | **0.99** | Best performing model |

**Top predictive factors:** `model_year`, `base_msrp`, and `electric_vehicle_type`.

## 📈 Visual Insights
- **Feature Importance:** `model_year` dominates as the strongest predictor of range.  
- **Residual Analysis:** Model residuals are centered and stable — minimal bias.  
- **Trend Plots:** Average EV range and registration counts both increase across years.

## 🧰 Tools & Libraries
- Python  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn (Pipeline, ColumnTransformer, RandomForestRegressor)

## 🧾 Dataset
Dataset: *Washington State EV Population Data (Cleaned version)*  
Source: Washington State Department of Licensing (DOL) Open Data Portal  
File used: `EV_Cleaned.csv`, Download from (Download Cleaned Dataset (Google Drive) Link: https://drive.google.com/file/d/1PF14pQv7424hyIW2XWU6hY6YE8NzsNdx/view?usp=sharing
