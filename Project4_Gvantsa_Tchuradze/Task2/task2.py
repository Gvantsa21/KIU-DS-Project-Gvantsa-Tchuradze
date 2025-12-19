
# Task 2: Model Training & Validation
# Project 4 – Introduction to Data Science with Python
# In this task we train different regression models and compare their performance

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

# Load the processed datasets created in Task 1
# These datasets are already cleaned, encoded, and scaled
X_train = pd.read_csv("Project4_Gvantsa_Tchuradze/Task1/X_train_processed.csv")
X_test = pd.read_csv("Project4_Gvantsa_Tchuradze/Task1/X_test_processed.csv")
y_train = pd.read_csv("Project4_Gvantsa_Tchuradze/Task1/y_train.csv").squeeze()
y_test = pd.read_csv("Project4_Gvantsa_Tchuradze/Task1/y_test.csv").squeeze()

print("Data loaded successfully")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# This function is used to evaluate models in the same way
# It returns train R2, test R2, MAE and RMSE
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)

    train_r2 = r2_score(y_tr, y_train_pred)
    test_r2 = r2_score(y_te, y_test_pred)
    mae = mean_absolute_error(y_te, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_test_pred))

    return train_r2, test_r2, mae, rmse

# This list will store results from all models
results = []

# Baseline model: Linear Regression
# This model is used as a simple reference to compare other models
print("\nLinear Regression (Baseline Model)")

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

lr_train_r2, lr_test_r2, lr_mae, lr_rmse = evaluate_model(
    linear_model, X_train, X_test, y_train, y_test
)

print("Train R2:", lr_train_r2)
print("Test R2:", lr_test_r2)
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)

# Cross-validation helps check how stable the model is
lr_cv_scores = cross_val_score(
    linear_model, X_train, y_train, cv=5, scoring="r2"
)

results.append([
    "Linear Regression",
    lr_train_r2,
    lr_test_r2,
    lr_cv_scores.mean(),
    lr_cv_scores.std(),
    lr_mae,
    lr_rmse
])

# Ridge Regression adds regularization to reduce overfitting
print("\nRidge Regression")

ridge_alphas = [0.1, 1, 10, 100]
best_alpha = None
best_cv_score = -np.inf

# Try different alpha values and select the one with best CV score
for alpha in ridge_alphas:
    ridge = Ridge(alpha=alpha)
    cv_scores = cross_val_score(
        ridge, X_train, y_train, cv=5, scoring="r2"
    )
    print("alpha =", alpha, "CV mean R2 =", cv_scores.mean())

    if cv_scores.mean() > best_cv_score:
        best_cv_score = cv_scores.mean()
        best_alpha = alpha

print("Best alpha selected:", best_alpha)

ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X_train, y_train)

ridge_train_r2, ridge_test_r2, ridge_mae, ridge_rmse = evaluate_model(
    ridge_best, X_train, X_test, y_train, y_test
)

ridge_cv_scores = cross_val_score(
    ridge_best, X_train, y_train, cv=5, scoring="r2"
)

results.append([
    "Ridge Regression",
    ridge_train_r2,
    ridge_test_r2,
    ridge_cv_scores.mean(),
    ridge_cv_scores.std(),
    ridge_mae,
    ridge_rmse
])

# Lasso Regression can set some coefficients to zero
# This helps with feature selection
print("\nLasso Regression")

lasso_alphas = [0.01, 0.1, 1, 10]
best_alpha = None
best_cv_score = -np.inf

for alpha in lasso_alphas:
    lasso = Lasso(alpha=alpha, max_iter=5000)
    cv_scores = cross_val_score(
        lasso, X_train, y_train, cv=5, scoring="r2"
    )
    print("alpha =", alpha, "CV mean R2 =", cv_scores.mean())

    if cv_scores.mean() > best_cv_score:
        best_cv_score = cv_scores.mean()
        best_alpha = alpha

print("Best alpha selected:", best_alpha)

lasso_best = Lasso(alpha=best_alpha, max_iter=5000)
lasso_best.fit(X_train, y_train)

lasso_train_r2, lasso_test_r2, lasso_mae, lasso_rmse = evaluate_model(
    lasso_best, X_train, X_test, y_train, y_test
)

# Count how many features were removed by Lasso
zero_coefficients = np.sum(lasso_best.coef_ == 0)
print("Number of zero coefficients:", zero_coefficients)

lasso_cv_scores = cross_val_score(
    lasso_best, X_train, y_train, cv=5, scoring="r2"
)

results.append([
    "Lasso Regression",
    lasso_train_r2,
    lasso_test_r2,
    lasso_cv_scores.mean(),
    lasso_cv_scores.std(),
    lasso_mae,
    lasso_rmse
])

# Decision Tree Regression
# First we limit depth to reduce overfitting
print("\nDecision Tree Regressor")

tree_limited = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_limited.fit(X_train, y_train)

tree_train_r2, tree_test_r2, tree_mae, tree_rmse = evaluate_model(
    tree_limited, X_train, X_test, y_train, y_test
)

tree_cv_scores = cross_val_score(
    tree_limited, X_train, y_train, cv=5, scoring="r2"
)

results.append([
    "Decision Tree (max_depth=5)",
    tree_train_r2,
    tree_test_r2,
    tree_cv_scores.mean(),
    tree_cv_scores.std(),
    tree_mae,
    tree_rmse
])

# Train an unrestricted tree to show overfitting
tree_full = DecisionTreeRegressor(random_state=42)
tree_full.fit(X_train, y_train)

print("Unrestricted tree train R2:",
      r2_score(y_train, tree_full.predict(X_train)))
print("Unrestricted tree test R2:",
      r2_score(y_test, tree_full.predict(X_test)))

# Random Forest Regressor (bonus model)
# It combines many trees to improve performance
print("\nRandom Forest Regressor")

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_train_r2, rf_test_r2, rf_mae, rf_rmse = evaluate_model(
    rf, X_train, X_test, y_train, y_test
)

rf_cv_scores = cross_val_score(
    rf, X_train, y_train, cv=5, scoring="r2"
)

results.append([
    "Random Forest",
    rf_train_r2,
    rf_test_r2,
    rf_cv_scores.mean(),
    rf_cv_scores.std(),
    rf_mae,
    rf_rmse
])

# Feature importance helps understand which features matter most
feature_importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 10 most important features (Random Forest):")
print(feature_importance.head(10))

# Create a comparison table for all models
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Train R2",
        "Test R2",
        "CV Mean R2",
        "CV Std R2",
        "MAE",
        "RMSE"
    ]
)

print("\nModel comparison table:")
print(results_df)

# Identify the best model based on test R2 score
best_model = results_df.sort_values(
    by="Test R2", ascending=False
).iloc[0]

print("\nBest performing model on test data:")
print(best_model)

print("\nTask 2 completed successfully.")
