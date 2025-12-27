import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set consistent visualization style for all plots
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Load the customer dataset
df = pd.read_csv("C:\\Users\\GVANTSA\\OneDrive\\Desktop\\datasc\\Project5_Gvantsa_Tchuradze\\customer_data.csv")

# Display basic dataset information to understand structure
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)

# Show descriptive statistics for numerical features
print("\nStatistical Summary:")
print(df.describe())

# Identify missing values in each column
print("\nMissing Values Per Column:")
print(df.isna().sum())

# Analyze the distribution of the target variable (customer churn)
print("\nChurn Distribution:")
print(df["Churned"].value_counts())

plt.figure()
df["Churned"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.xlabel("Churned (0 = Active, 1 = Churned)")
plt.ylabel("Count")
plt.show()

# Visualize distributions of numerical features to detect skewness and outliers
numerical_features = [
    "Age",
    "Annual_Income",
    "Total_Amount_Spent",
    "Purchase_Frequency",
    "Days_Since_Last_Purchase",
    "Satisfaction_Score"
]

for col in numerical_features:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Compute correlations between numerical variables
plt.figure(figsize=(14, 10))
corr = df.select_dtypes(include="number").corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

# Compare key behavioral features between churned and active customers
features_to_compare = [
    "Total_Amount_Spent",
    "Purchase_Frequency",
    "Days_Since_Last_Purchase",
    "Satisfaction_Score"
]

for col in features_to_compare:
    plt.figure()
    sns.boxplot(x="Churned", y=col, data=df)
    plt.title(f"{col} vs Churn")
    plt.show()

# Identify the strongest correlations with churn
churn_corr = corr["Churned"].abs().sort_values(ascending=False)
print("\nTop 5 Features Most Correlated with Churn:")
print(churn_corr[1:6])

# Analyze membership type distribution
plt.figure()
df["Membership_Type"].value_counts().plot(kind="bar")
plt.title("Membership Type Distribution")
plt.show()

# Examine churn behavior across different membership types
plt.figure()
sns.countplot(x="Membership_Type", hue="Churned", data=df)
plt.title("Membership Type vs Churn")
plt.show()

# Compare engagement levels across location types
plt.figure()
sns.boxplot(x="Location_Type", y="Visits_Per_Month", data=df)
plt.title("Visits Per Month by Location Type")
plt.show()

"""
Key Insights:
- Customers with lower purchase frequency and longer inactivity periods are more likely to churn.
- Lower satisfaction scores are strongly associated with churn.
- VIP members exhibit significantly lower churn rates.
- Urban customers show higher engagement compared to other locations.

Hypothesis:
Customer engagement and satisfaction are major drivers of churn,
while loyalty programs reduce churn probability.
"""

# Fill missing numerical values using the median to reduce the effect of outliers
numerical_cols = df.select_dtypes(include="number").columns
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Remove identifiers and derived risk score to avoid data leakage
df = df.drop(columns=["Customer_ID", "Churn_Risk_Score"])

# Convert categorical variables into numerical format using one-hot encoding
categorical_cols = [
    "Gender",
    "Education",
    "Location_Type",
    "Membership_Type",
    "Payment_Method",
    "Favorite_Category"
]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features from the target variable
X = df_encoded.drop("Churned", axis=1)
y = df_encoded["Churned"]

# Standardize features to improve model performance
scaler = StandardScaler()

# Split data into training and testing sets while preserving class balance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply scaling to training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Final verification of preprocessing steps
print("\nPreprocessing Complete")
print("Training Set Shape:", X_train_scaled.shape)
print("Test Set Shape:", X_test_scaled.shape)
print("\nClass Balance (Training Set):")
print(y_train.value_counts(normalize=True))
print("\nTask 1 Successfully Completed!")
