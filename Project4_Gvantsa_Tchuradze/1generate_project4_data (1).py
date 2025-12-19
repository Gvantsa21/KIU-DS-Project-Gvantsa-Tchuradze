"""
Project 4: Machine Learning Pipeline - Data Generator
Kutaisi International University
Introduction to Data Science with Python

This script generates realistic house price data for practicing
regression modeling and machine learning pipelines.

Instructions:
1. Run this script to generate the CSV file
2. Use this file for all tasks in Project 4
3. Do not modify this script or the generated CSV file before starting your work

Generated file:
- house_prices.csv (800 houses with features and prices)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 70)
print("Project 4: Machine Learning Pipeline - Data Generator")
print("Introduction to Data Science with Python")
print("=" * 70)
print()

# ============================================================================
# PART 1: Generate House Features
# ============================================================================

print("[1/6] Generating basic house features...")

n_houses = 800

# House ID
house_ids = [f"H{str(i + 1).zfill(4)}" for i in range(n_houses)]

# Location features
neighborhoods = ["Downtown", "Suburban", "Rural", "Beachfront", "Mountain"]
neighborhood = np.random.choice(
    neighborhoods, n_houses, p=[0.25, 0.35, 0.20, 0.12, 0.08]
)

# Distance from city center (km)
distance_city = np.random.exponential(8, n_houses)
distance_city = np.clip(distance_city, 0.5, 40)

# House characteristics
square_footage = np.random.normal(2000, 600, n_houses)
square_footage = np.clip(square_footage, 800, 5000).astype(int)

num_bedrooms = np.random.choice(
    [1, 2, 3, 4, 5, 6], n_houses, p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03]
)
num_bathrooms = num_bedrooms * 0.75 + np.random.uniform(-0.5, 0.5, n_houses)
num_bathrooms = np.clip(num_bathrooms, 1, 5)

# Lot size (square meters)
lot_size = square_footage * np.random.uniform(1.5, 4, n_houses)
lot_size = lot_size.astype(int)

print("[2/6] Generating house quality features...")

# Year built
year_built = np.random.normal(1995, 20, n_houses)
year_built = np.clip(year_built, 1950, 2024).astype(int)

# Age of house
house_age = 2024 - year_built

# Condition (1-5 scale)
# Newer houses tend to be in better condition
condition_base = 5 - (house_age / 75 * 3) + np.random.normal(0, 0.8, n_houses)
condition = np.clip(condition_base, 1, 5).round()

# Renovation status
last_renovated_prob = np.where(house_age > 15, 0.4, 0.1)
has_been_renovated = np.random.random(n_houses) < last_renovated_prob
years_since_renovation = np.where(
    has_been_renovated, np.random.uniform(0, 10, n_houses), house_age
)

# Number of floors
num_floors = np.random.choice([1, 2, 3], n_houses, p=[0.40, 0.50, 0.10])

# Garage
garage_spaces = np.random.choice([0, 1, 2, 3], n_houses, p=[0.10, 0.30, 0.45, 0.15])

print("[3/6] Generating amenities and features...")

# Amenities (binary features)
has_pool = np.random.choice([0, 1], n_houses, p=[0.75, 0.25])
has_garden = np.random.choice([0, 1], n_houses, p=[0.35, 0.65])
has_basement = np.random.choice([0, 1], n_houses, p=[0.60, 0.40])
has_attic = np.random.choice([0, 1], n_houses, p=[0.70, 0.30])

# Heating/Cooling
heating_type = np.random.choice(
    ["Gas", "Electric", "Oil", "Solar"], n_houses, p=[0.50, 0.30, 0.15, 0.05]
)
has_ac = np.random.choice([0, 1], n_houses, p=[0.25, 0.75])

# Energy efficiency rating (A-G)
energy_ratings = ["A", "B", "C", "D", "E", "F", "G"]
energy_rating = np.random.choice(
    energy_ratings, n_houses, p=[0.08, 0.15, 0.25, 0.30, 0.15, 0.05, 0.02]
)

print("[4/6] Generating location-specific features...")

# School rating (1-10) - better in suburban areas
school_rating_base = np.where(
    neighborhood == "Suburban",
    np.random.normal(7.5, 1.5, n_houses),
    np.random.normal(6, 2, n_houses),
)
school_rating = np.clip(school_rating_base, 1, 10)

# Crime rate (per 1000 residents)
crime_rate_base = np.where(
    neighborhood == "Downtown",
    np.random.exponential(15, n_houses),
    np.random.exponential(8, n_houses),
)
crime_rate = np.clip(crime_rate_base, 1, 50)

# Walkability score (0-100)
walkability_base = np.where(
    neighborhood == "Downtown",
    np.random.normal(85, 10, n_houses),
    np.where(
        neighborhood == "Suburban",
        np.random.normal(60, 15, n_houses),
        np.random.normal(30, 15, n_houses),
    ),
)
walkability_score = np.clip(walkability_base, 0, 100)

# Public transport access (0-10)
public_transport_base = (
    10 - (distance_city / 40 * 8) + np.random.normal(0, 1.5, n_houses)
)
public_transport = np.clip(public_transport_base, 0, 10)

print("[5/6] Calculating house prices...")

# ============================================================================
# PART 2: Calculate Target Variable (Price)
# ============================================================================

# Base price calculation with realistic relationships
base_price = (
    # Square footage is primary driver
    square_footage * 150
    +
    # Bedrooms/Bathrooms
    num_bedrooms * 25000
    + num_bathrooms * 15000
    +
    # Location factors
    np.where(neighborhood == "Beachfront", 200000, 0)
    + np.where(neighborhood == "Downtown", 100000, 0)
    + np.where(neighborhood == "Mountain", 80000, 0)
    + np.where(neighborhood == "Suburban", 40000, 0)
    +
    # Distance penalty
    -distance_city * 3000
    +
    # Age depreciation
    -house_age * 2000
    +
    # Condition bonus
    condition * 30000
    +
    # Renovation bonus
    np.where(has_been_renovated, 50000, 0)
    +
    # Amenities
    has_pool * 40000
    + has_garden * 15000
    + has_basement * 25000
    + garage_spaces * 20000
    +
    # Lot size
    lot_size * 20
    +
    # School rating impact
    school_rating * 8000
    +
    # Crime rate penalty
    -crime_rate * 1000
    +
    # Walkability bonus
    walkability_score * 500
    +
    # Energy efficiency
    (ord("H") - ord(energy_rating[0])) * 5000
    +
    # Floors
    num_floors * 15000
)

# Add realistic noise
price_noise = np.random.normal(0, 30000, n_houses)
price = base_price + price_noise

# Ensure reasonable price range
price = np.clip(price, 50000, 1500000)

print("[6/6] Creating dataset and adding data quality issues...")

# ============================================================================
# PART 3: Create DataFrame and Introduce Issues
# ============================================================================

data = {
    "House_ID": house_ids,
    "Neighborhood": neighborhood,
    "Distance_City_km": np.round(distance_city, 1),
    "Square_Footage": square_footage,
    "Lot_Size": lot_size,
    "Num_Bedrooms": num_bedrooms,
    "Num_Bathrooms": np.round(num_bathrooms, 1),
    "Num_Floors": num_floors,
    "Year_Built": year_built,
    "House_Age": house_age,
    "Condition": condition.astype(int),
    "Has_Been_Renovated": has_been_renovated.astype(int),
    "Years_Since_Renovation": np.round(years_since_renovation, 1),
    "Garage_Spaces": garage_spaces,
    "Has_Pool": has_pool,
    "Has_Garden": has_garden,
    "Has_Basement": has_basement,
    "Has_Attic": has_attic,
    "Heating_Type": heating_type,
    "Has_AC": has_ac,
    "Energy_Rating": energy_rating,
    "School_Rating": np.round(school_rating, 1),
    "Crime_Rate": np.round(crime_rate, 1),
    "Walkability_Score": np.round(walkability_score, 0).astype(int),
    "Public_Transport": np.round(public_transport, 1),
    "Price": np.round(price, 0).astype(int),
}

df = pd.DataFrame(data)

# Introduce some missing values (realistic for ML projects)
missing_columns = [
    "Years_Since_Renovation",
    "School_Rating",
    "Crime_Rate",
    "Energy_Rating",
]
for col in missing_columns:
    if col == "Years_Since_Renovation":
        # Only houses that haven't been renovated should have missing values
        mask = df["Has_Been_Renovated"] == 0
        df.loc[mask, col] = np.nan
    else:
        missing_indices = np.random.choice(
            df.index, size=int(len(df) * 0.05), replace=False
        )
        df.loc[missing_indices, col] = np.nan

# Add a few outliers (data entry errors)
outlier_indices = np.random.choice(df.index, size=5, replace=False)
df.loc[outlier_indices, "Price"] = df.loc[outlier_indices, "Price"] * np.random.uniform(
    2, 3, 5
)

# ============================================================================
# PART 4: Save and Report
# ============================================================================

print()
print("=" * 70)
print("Saving CSV file...")
print("=" * 70)

df.to_csv("house_prices.csv", index=False)
print(f"✓ house_prices.csv saved ({len(df)} houses)")

print()
print("=" * 70)
print("Dataset Summary")
print("=" * 70)
print()
print(f"Total Houses: {len(df)}")
print(f"Features: {len(df.columns) - 2}")  # Excluding ID and Price
print(f"Target Variable: Price")
print()

print("PRICE STATISTICS:")
print(f"  • Mean Price: ${df['Price'].mean():,.0f}")
print(f"  • Median Price: ${df['Price'].median():,.0f}")
print(f"  • Price Range: ${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")
print(f"  • Standard Deviation: ${df['Price'].std():,.0f}")
print()

print("FEATURE TYPES:")
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove("Price")
if "House_ID" in numerical_features:
    numerical_features.remove("House_ID")
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
if "House_ID" in categorical_features:
    categorical_features.remove("House_ID")

print(f"  • Numerical Features: {len(numerical_features)}")
print(f"  • Categorical Features: {len(categorical_features)}")
print(
    f"  • Binary Features: {(df[numerical_features] == 0).sum().sum() + (df[numerical_features] == 1).sum().sum()}"
)
print()

print("NEIGHBORHOOD DISTRIBUTION:")
for hood in neighborhoods:
    count = (df["Neighborhood"] == hood).sum()
    avg_price = df[df["Neighborhood"] == hood]["Price"].mean()
    print(f"  • {hood}: {count} houses (avg ${avg_price:,.0f})")
print()

print("MISSING VALUES:")
missing_count = df.isnull().sum()
if missing_count.sum() > 0:
    for col in missing_count[missing_count > 0].index:
        print(
            f"  • {col}: {missing_count[col]} missing ({missing_count[col] / len(df) * 100:.1f}%)"
        )
else:
    print("  • No missing values")
print()

print("KEY CORRELATIONS TO EXPLORE:")
print("  • Square Footage vs Price")
print("  • Location (Neighborhood) vs Price")
print("  • House Age vs Price")
print("  • School Rating vs Price")
print("  • Amenities vs Price")
print("  • Crime Rate vs Price")
print()

print("=" * 70)
print("Machine Learning Tasks:")
print("=" * 70)
print()
print("REGRESSION TARGET:")
print("  • Predict: House Price (continuous variable)")
print()
print("FEATURE ENGINEERING OPPORTUNITIES:")
print("  • Price per square foot")
print("  • Total rooms (bedrooms + bathrooms)")
print("  • Luxury score (pool + AC + garage)")
print("  • Location quality index")
print("  • Age categories")
print()
print("PREPROCESSING REQUIRED:")
print("  • Handle missing values")
print("  • Encode categorical variables (Neighborhood, Heating_Type, Energy_Rating)")
print("  • Scale numerical features")
print("  • Remove or handle outliers")
print("  • Split train/test sets")
print()
print("MODELS TO TRY:")
print("  • Linear Regression (baseline)")
print("  • Ridge/Lasso Regression (regularization)")
print("  • Decision Tree Regressor")
print("  • Random Forest Regressor")
print()
print("EVALUATION METRICS:")
print("  • R² Score")
print("  • Mean Absolute Error (MAE)")
print("  • Mean Squared Error (MSE)")
print("  • Root Mean Squared Error (RMSE)")
print()
print("=" * 70)
print("Data generation complete!")
print("You can now proceed with Project 4: Machine Learning Pipeline")
print("=" * 70)
print()
print("NOTE: Do NOT modify this CSV file before starting your project.")
print("      All preprocessing should be done in your project code.")
print("=" * 70)
