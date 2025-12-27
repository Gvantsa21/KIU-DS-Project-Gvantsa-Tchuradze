import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Set a consistent visualization style
sns.set(style="whitegrid")

# Load the dataset
df = pd.read_csv("C:\\Users\\GVANTSA\\OneDrive\\Desktop\\datasc\\Project5_Gvantsa_Tchuradze\\customer_data.csv")

# Fill missing numerical values using the median to reduce bias from outliers
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Remove identifier and leakage-prone columns
df = df.drop(columns=["Customer_ID", "Churn_Risk_Score"])

# Convert categorical variables into numerical format using one-hot encoding
df = pd.get_dummies(
    df,
    columns=[
        "Gender",
        "Education",
        "Location_Type",
        "Membership_Type",
        "Payment_Method",
        "Favorite_Category"
    ],
    drop_first=True
)

# Separate predictors from the target variable
X = df.drop("Churned", axis=1)
y = df["Churned"]

# Split the data into training and test sets while preserving class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardize features to ensure fair contribution across models
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model as a baseline classifier
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]

print("\nLOGISTIC REGRESSION")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Train a decision tree to capture non-linear relationships
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

print("\nDECISION TREE")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Examine feature importance from the decision tree
dt_importance = pd.Series(dt.feature_importances_, index=X.columns)
print("\nTop Decision Tree Features:")
print(dt_importance.sort_values(ascending=False).head(10))

# Train a random forest model for improved performance and robustness
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("\nRANDOM FOREST")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Train a support vector machine to model complex decision boundaries
svm = SVC(kernel="rbf", probability=True)
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:, 1]

print("\nSUPPORT VECTOR MACHINE")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Retrain logistic regression with class balancing to address class imbalance
log_reg_bal = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
log_reg_bal.fit(X_train, y_train)

y_pred_lr_bal = log_reg_bal.predict(X_test)
y_prob_lr_bal = log_reg_bal.predict_proba(X_test)[:, 1]

print("\nBALANCED LOGISTIC REGRESSION")
print(confusion_matrix(y_test, y_pred_lr_bal))
print(classification_report(y_test, y_pred_lr_bal))

# Define a unified evaluation function for fair model comparison
def evaluate_model(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }

# Compile performance metrics for all models
results = pd.DataFrame(
    [
        evaluate_model(y_test, y_pred_lr, y_prob_lr),
        evaluate_model(y_test, y_pred_dt, y_prob_dt),
        evaluate_model(y_test, y_pred_rf, y_prob_rf),
        evaluate_model(y_test, y_pred_svm, y_prob_svm),
        evaluate_model(y_test, y_pred_lr_bal, y_prob_lr_bal)
    ],
    index=[
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "Balanced Logistic Regression"
    ]
)

print("\nMODEL COMPARISON")
print(results)

# Plot ROC curves to visually compare classifier performance
plt.figure(figsize=(8, 6))

models = {
    "Logistic Regression": y_prob_lr,
    "Decision Tree": y_prob_dt,
    "Random Forest": y_prob_rf,
    "SVM": y_prob_svm,
    "Balanced Logistic Regression": y_prob_lr_bal
}

for name, probs in models.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Model Comparison")
plt.legend()
plt.show()

# Visualize the most influential features from the best-performing model
rf_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
rf_importance.plot(kind="bar")
plt.title("Top 10 Features Driving Customer Churn")
plt.ylabel("Importance")
plt.show()

"""
Interpretation:
- Time since last purchase is the strongest indicator of churn
- Lower spending and purchase frequency increase churn likelihood
- Satisfaction and engagement metrics strongly influence retention
- Loyalty-related features reduce churn probability

Best Model:
Random Forest achieves the best overall performance due to
its high ROC-AUC and balanced precision–recall trade-off.
"""

print("\nTask 2 (Classification) Completed Successfully!")
