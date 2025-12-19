# 🤖 Project 4 — Machine Learning Pipeline  
**Introduction to Data Science with Python**
**Name:** Gvantsa Tchuradze
**Student ID:** 
**Date:** 12/19/2025 
**Honor Code:** *I certify that this work is my own.*
---

## 📌 Project Overview

The goal of this project is to design and implement a **complete machine learning pipeline** for predicting house prices.

The project follows a structured data science workflow including:
- Data preprocessing and feature engineering
- Model training and comparison
- Model evaluation, improvement, and interpretation

The final outcome is a **well-evaluated and tuned predictive model**, supported by visualizations, performance metrics, and a detailed analytical report.

---

## 📂 Project Structure

Project4_Gvantsa_Tchuradze/
│
├── Task1/
│ ├── X_train_processed.csv
│ ├── X_test_processed.csv
│ ├── y_train.csv
│ └── y_test.csv
│
├── Task2/
│ └── model_comparison_results.csv
│
├── Task3/
│ ├── residual_analysis.png
│ ├── feature_importance.png
│ ├── model_evaluation_report.txt
│ └── final_tuned_model.pkl
│
├── task1_data_preprocessing.py
├── task2_model_training.py
├── task3_model_evaluation.py
│
└── README.md



---

## 🧩 Task 1 — Data Preprocessing & Feature Engineering

### 🎯 Objective
Prepare the raw dataset for machine learning by:
- Cleaning missing values
- Encoding categorical variables
- Scaling numerical features
- Engineering new informative features

### 🔑 Key Steps
- Train–test split to prevent data leakage
- Numerical feature scaling
- One-hot encoding for categorical variables
- Feature engineering, including:
  - `Total_Rooms`
  - `Is_New`
  - `Luxury_Score`
  - `Location_Score`

### 📤 Outputs
- `X_train_processed.csv`
- `X_test_processed.csv`
- `y_train.csv`
- `y_test.csv`

These processed datasets are reused consistently in **Task 2** and **Task 3**.

---

## 🧠 Task 2 — Model Training & Comparison

### 🎯 Objective
Train multiple regression models and identify the best-performing one.

### 🤖 Models Evaluated
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor

### 📏 Evaluation Metrics
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### 🏆 Result
The **Random Forest Regressor** achieved the best balance between bias and variance and demonstrated superior predictive performance on the test set.

### 📤 Output
- `model_comparison_results.csv`

---

## 📊 Task 3 — Model Evaluation & Improvement

Task 3 performs **in-depth evaluation, tuning, and interpretation** of the best model selected in Task 2.

---

### 🔹 Part A — Detailed Model Evaluation
- Retrained Random Forest on the full training set
- Computed performance metrics:
  - Train R²
  - Test R²
  - MAE
  - RMSE
- Conducted residual analysis:
  - Predicted vs Actual plot
  - Residuals vs Predicted values
  - Residual histogram
  - Q–Q plot

#### 📁 Output
- `residual_analysis.png`

---

### 🔹 Part B — Hyperparameter Tuning
- Applied `GridSearchCV` with **5-fold cross-validation**
- Tuned hyperparameters:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
- Achieved measurable improvements in:
  - Test R²
  - RMSE

---

### 🔹 Part C — Feature Importance Analysis
- Extracted feature importance from the tuned Random Forest
- Identified top predictors of house prices
- Verified the effectiveness of engineered features
- Visualized the top 10 most important features

#### 📁 Output
- `feature_importance.png`

---

### 🔹 Part D — Final Report & Model Saving (Bonus)
- Generated a comprehensive textual report covering:
  - Best model performance
  - Feature insights
  - Model limitations
  - Business recommendations
- Saved the final trained model for future use

#### 📁 Outputs
- `model_evaluation_report.txt`
- `final_tuned_model.pkl`

---

## 📈 Final Model Performance (After Tuning)

- **Model:** Random Forest Regressor  
- **Explained Variance:** ~90% of house price variability  
- **Average Error:** Within a reasonable percentage range  
- **Generalization:** Minimal overfitting (similar train & test R²)

---

## 💡 Key Insights
- Property size, location quality, and amenities are the strongest price drivers
- Feature engineering significantly improved model performance
- Random Forest handled non-linear relationships and noise effectively
- Prediction errors are larger for extreme-priced properties

---

## ⚠️ Limitations & Future Improvements

### Limitations
- Reduced accuracy for very high or very low priced houses
- Slight heteroscedasticity observed in residuals

### Future Improvements
- Gradient boosting models (XGBoost, LightGBM)
- Additional location-based features
- Temporal market trend analysis

---

## ✅ Conclusion

This project successfully demonstrates a **complete machine learning pipeline**, from raw data preprocessing to final model deployment.

Through systematic evaluation, hyperparameter tuning, and feature interpretation, a **robust and reliable house price prediction model** was developed.

All required and bonus components of **Project 4** have been successfully completed.
