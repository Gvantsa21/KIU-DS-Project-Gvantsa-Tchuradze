
# Task 1: Data Preprocessing & Feature Engineering
# Introduction to Data Science with Python – Project 4

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Part A: Data Loading & Exploration

# Load dataset"
df = pd.read_csv("Project4_Gvantsa_Tchuradze/house_prices.csv")

# Basic dataset info
print("Dataset shape:", df.shape)
print("\nColumn names and data types:")
print(df.dtypes)

# Statistical summary of numerical columns
print("\nStatistical summary:")
print(df.describe())

# Missing values per column
print("\nMissing values per column:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicate rows:", df.duplicated().sum())


# Exploratory Analysis

# Target variable distribution
print("\nPrice distribution:")
print(df["Price"].describe())

price_mean = df["Price"].mean()
price_median = df["Price"].median()
price_std = df["Price"].std()

print("Price mean:", round(price_mean, 2))
print("Price median:", round(price_median, 2))
print("Price std:", round(price_std, 2))

# Identify numerical and categorical features
num_features = df.select_dtypes(include=[np.number]).columns.tolist()
cat_features = df.select_dtypes(include=["object"]).columns.tolist()

# remove target from numerical features
num_features.remove("Price")

# remove ID from categorical features
cat_features.remove("House_ID")


print("\nNumerical features:", num_features)
print("\nCategorical features:", cat_features)

# Identify outliers in Price (> 3 std)
upper = price_mean + 3 * price_std
lower = price_mean - 3 * price_std

outliers = df[(df["Price"] > upper) | (df["Price"] < lower)]
print("\nNumber of price outliers:", outliers.shape[0])

# Correlation with target
print("\nTop correlations with Price:")
print(df.corr(numeric_only=True)["Price"].sort_values(ascending=False).head(15))





# Part B: Data Preprocessing

# Handle missing values

# Years_Since_Renovation:
# If house not renovated, fill with House_Age
mask = (df["Has_Been_Renovated"] == 0) & (df["Years_Since_Renovation"].isnull())
df.loc[mask, "Years_Since_Renovation"] = df.loc[mask, "House_Age"]

# Fill remaining Years_Since_Renovation with median
df["Years_Since_Renovation"] = df["Years_Since_Renovation"].fillna(
    df["Years_Since_Renovation"].median()
)

# School_Rating: fill with median
df["School_Rating"] = df["School_Rating"].fillna(df["School_Rating"].median())

# Crime_Rate: fill with median
df["Crime_Rate"] = df["Crime_Rate"].fillna(df["Crime_Rate"].median())

# Energy_Rating: fill with mode
df["Energy_Rating"] = df["Energy_Rating"].fillna(df["Energy_Rating"].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())


# Handle Outliers (Price)

# Cap prices at ±3 standard deviations to keep data size
df["Price_original"] = df["Price"]
df["Price"] = df["Price"].clip(lower=lower, upper=upper)

print("\nNumber of capped prices:",
      (df["Price"] != df["Price_original"]).sum())


# Encode Categorical Variables

# Energy_Rating mapping: A=7, B=6, ..., G=1
energy_map = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
df["Energy_Rating_Num"] = df["Energy_Rating"].map(energy_map)

# One-hot encode Neighborhood and Heating_Type
df = pd.concat([df, pd.get_dummies(df["Neighborhood"], prefix="NH")], axis=1)
df = pd.concat([df, pd.get_dummies(df["Heating_Type"], prefix="Heat")], axis=1)




# Part C: Feature Engineering

# Total number of rooms
df["Total_Rooms"] = df["Num_Bedrooms"] + df["Num_Bathrooms"]

# New house indicator
df["Is_New"] = (df["House_Age"] < 5).astype(int)

# Luxury score
df["Luxury_Score"] = (
    df["Has_Pool"] +
    df["Has_AC"] +
    (df["Garage_Spaces"] > 1).astype(int)
)

# Location score: School quality minus crime
df["School_norm"] = (
    (df["School_Rating"] - df["School_Rating"].min()) /
    (df["School_Rating"].max() - df["School_Rating"].min())
)

df["Crime_norm"] = (
    (df["Crime_Rate"] - df["Crime_Rate"].min()) /
    (df["Crime_Rate"].max() - df["Crime_Rate"].min())
)

df["Location_Score"] = df["School_norm"] - df["Crime_norm"]

# Price per square foot (EDA only, not for modeling)
df["Price_Per_SqFt_EDA"] = df["Price_original"] / df["Square_Footage"]





# Part D: Train-Test Split

# Prepare features and target
exclude_cols = [
    "House_ID", "Neighborhood", "Heating_Type",
    "Energy_Rating", "Price_original", "Price"
]

X = df.drop(columns=exclude_cols)
X = X.select_dtypes(include=[np.number])
y = df["Price"]

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTrain/test shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# -------------------------
# Feature Scaling
# -------------------------

# Scale numeric features using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Save scaler for later use
joblib.dump(scaler, "standard_scaler.pkl")

# Final checks
print("\nFinal checks:")
print("Missing values in X_train:", X_train_scaled.isnull().sum().sum())
print("Missing values in X_test:", X_test_scaled.isnull().sum().sum())

# Save processed datasets
X_train_scaled.to_csv("X_train_processed.csv", index=False)
X_test_scaled.to_csv("X_test_processed.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nTask 1 preprocessing completed successfully.")
