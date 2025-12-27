"""
Project 5: Advanced Analytics - Data Generator
Kutaisi International University
Introduction to Data Science with Python

This script generates realistic customer data for e-commerce platform
suitable for both classification (churn prediction) and clustering (segmentation).

Instructions:
1. Run this script to generate the CSV file
2. Choose either Classification or Clustering approach (or both!)
3. Do not modify this script or the generated CSV file before starting your work

Generated file:
- customer_data.csv (1000 customers with behavioral and demographic features)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 70)
print("Project 5: Advanced Analytics - Data Generator")
print("Introduction to Data Science with Python")
print("=" * 70)
print()

# ============================================================================
# PART 1: Generate Customer Demographics
# ============================================================================

print("[1/7] Generating customer demographics...")

n_customers = 1000

# Customer ID
customer_ids = [f"CUST{str(i + 1).zfill(4)}" for i in range(n_customers)]

# Demographics
ages = np.random.normal(40, 15, n_customers)
ages = np.clip(ages, 18, 80).astype(int)

gender = np.random.choice(
    ["Male", "Female", "Other"], n_customers, p=[0.48, 0.48, 0.04]
)

# Income (correlated with age - peaks mid-career)
income_base = 30000 + (ages - 18) * 1500 - ((ages - 45) ** 2) * 10
income_noise = np.random.normal(0, 15000, n_customers)
annual_income = np.clip(income_base + income_noise, 20000, 200000)

# Education level
education = np.random.choice(
    ["High School", "Bachelor", "Master", "PhD"],
    n_customers,
    p=[0.30, 0.45, 0.20, 0.05],
)

# Location type
location_type = np.random.choice(
    ["Urban", "Suburban", "Rural"], n_customers, p=[0.45, 0.40, 0.15]
)

print("[2/7] Generating account information...")

# ============================================================================
# PART 2: Account Information
# ============================================================================

# Account age (days)
account_age_days = np.random.exponential(400, n_customers)
account_age_days = np.clip(account_age_days, 30, 2000).astype(int)

# Membership type
membership_type = np.random.choice(
    ["Basic", "Premium", "VIP"], n_customers, p=[0.50, 0.35, 0.15]
)

# Payment method preference
payment_method = np.random.choice(
    ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"],
    n_customers,
    p=[0.40, 0.30, 0.20, 0.10],
)

print("[3/7] Generating purchase behavior...")

# ============================================================================
# PART 3: Purchase Behavior
# ============================================================================

# Total number of purchases (influenced by membership and account age)
base_purchases = (account_age_days / 30) * np.where(
    membership_type == "VIP", 3, np.where(membership_type == "Premium", 2, 1)
)
total_purchases = base_purchases + np.random.normal(0, 5, n_customers)
total_purchases = np.clip(total_purchases, 0, 200).astype(int)

# Total amount spent
avg_purchase_value = np.random.gamma(3, 30, n_customers)  # Varies per customer
total_amount_spent = total_purchases * avg_purchase_value
total_amount_spent = np.clip(total_amount_spent, 0, 50000)

# Average order value
avg_order_value = np.where(total_purchases > 0, total_amount_spent / total_purchases, 0)

# Days since last purchase
days_since_last_purchase = np.random.exponential(45, n_customers)
days_since_last_purchase = np.clip(days_since_last_purchase, 0, 365).astype(int)

# Purchase frequency (purchases per month)
months_active = account_age_days / 30
purchase_frequency = np.where(months_active > 0, total_purchases / months_active, 0)

print("[4/7] Generating engagement metrics...")

# ============================================================================
# PART 4: Engagement Metrics
# ============================================================================

# Website visits per month
visits_per_month = np.random.gamma(2, 5, n_customers)
visits_per_month = np.clip(visits_per_month, 0, 100)

# Time spent on site (minutes per visit)
time_per_visit = np.random.gamma(2, 3, n_customers)
time_per_visit = np.clip(time_per_visit, 1, 60)

# Product views per visit
product_views_per_visit = np.random.poisson(8, n_customers)
product_views_per_visit = np.clip(product_views_per_visit, 1, 50)

# Cart abandonment rate (%)
cart_abandonment_rate = np.random.beta(3, 3, n_customers) * 100
cart_abandonment_rate = np.clip(cart_abandonment_rate, 0, 100)

# Email open rate (%)
email_open_rate = np.random.beta(4, 6, n_customers) * 100
email_open_rate = np.clip(email_open_rate, 0, 100)

# Customer service contacts
service_contacts = np.random.poisson(2, n_customers)
service_contacts = np.clip(service_contacts, 0, 20)

print("[5/7] Generating satisfaction and loyalty metrics...")

# ============================================================================
# PART 5: Satisfaction and Loyalty
# ============================================================================

# Customer satisfaction score (1-10)
satisfaction_base = 7 + (total_purchases / 50) - (service_contacts * 0.5)
satisfaction_score = satisfaction_base + np.random.normal(0, 1.5, n_customers)
satisfaction_score = np.clip(satisfaction_score, 1, 10)

# Net Promoter Score (-100 to 100)
nps_base = (satisfaction_score - 5) * 20
nps = nps_base + np.random.normal(0, 15, n_customers)
nps = np.clip(nps, -100, 100)

# Number of referrals made
referrals = np.where(
    nps > 30, np.random.poisson(2, n_customers), np.random.poisson(0.3, n_customers)
)
referrals = np.clip(referrals, 0, 10)

# Loyalty points balance
loyalty_points = total_amount_spent * 0.1 + np.random.normal(0, 500, n_customers)
loyalty_points = np.clip(loyalty_points, 0, 10000).astype(int)

# Number of complaints
complaints = np.random.poisson(0.5, n_customers)
complaints = np.clip(complaints, 0, 10)

print("[6/7] Generating product preferences...")

# ============================================================================
# PART 6: Product Preferences
# ============================================================================

# Favorite category
categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Sports", "Beauty"]
favorite_category = np.random.choice(categories, n_customers)

# Number of categories shopped
num_categories = np.random.choice(
    [1, 2, 3, 4, 5, 6], n_customers, p=[0.20, 0.30, 0.25, 0.15, 0.08, 0.02]
)

# Discount usage rate (% of purchases with discount)
discount_usage_rate = np.random.beta(3, 5, n_customers) * 100
discount_usage_rate = np.clip(discount_usage_rate, 0, 100)

# Returns rate (% of purchases returned)
returns_rate = np.random.beta(2, 10, n_customers) * 100
returns_rate = np.clip(returns_rate, 0, 50)

print("[7/7] Calculating churn probability and status...")

# ============================================================================
# PART 7: Calculate Churn (Target Variable for Classification)
# ============================================================================

# Churn risk score (0-100) based on various factors
churn_score = (
    # Negative factors (increase churn)
    (days_since_last_purchase / 365 * 30)  # Inactive customers
    + (cart_abandonment_rate / 100 * 15)  # High abandonment
    + (complaints * 5)  # Many complaints
    + ((10 - satisfaction_score) * 3)  # Low satisfaction
    + (service_contacts * 2)  # High service needs
    + (returns_rate / 100 * 10)  # High returns
    -
    # Positive factors (decrease churn)
    (purchase_frequency * -5)  # Frequent buyers
    - (loyalty_points / 100 * -0.5)  # High loyalty
    - (referrals * -3)  # Advocates
    - (email_open_rate / 100 * -10)  # Engaged
    - (
        np.where(
            membership_type == "VIP", -15, np.where(membership_type == "Premium", -8, 0)
        )
    )
)

# Add noise
churn_score = churn_score + np.random.normal(0, 10, n_customers)
churn_score = np.clip(churn_score, 0, 100)

# Convert to binary churn status (threshold at 50)
churned = (churn_score > 50).astype(int)

# Add some randomness to make it realistic (not perfect separation)
churn_noise = np.random.random(n_customers)
churned = np.where(
    churn_noise < 0.1,  # 10% random flips
    1 - churned,
    churned,
)

# ============================================================================
# PART 8: Create DataFrame
# ============================================================================

print()
print("Creating comprehensive dataset...")

data = {
    "Customer_ID": customer_ids,
    "Age": ages,
    "Gender": gender,
    "Annual_Income": np.round(annual_income, 0).astype(int),
    "Education": education,
    "Location_Type": location_type,
    "Account_Age_Days": account_age_days,
    "Membership_Type": membership_type,
    "Payment_Method": payment_method,
    "Total_Purchases": total_purchases,
    "Total_Amount_Spent": np.round(total_amount_spent, 2),
    "Avg_Order_Value": np.round(avg_order_value, 2),
    "Days_Since_Last_Purchase": days_since_last_purchase,
    "Purchase_Frequency": np.round(purchase_frequency, 2),
    "Visits_Per_Month": np.round(visits_per_month, 1),
    "Time_Per_Visit_Minutes": np.round(time_per_visit, 1),
    "Product_Views_Per_Visit": product_views_per_visit,
    "Cart_Abandonment_Rate": np.round(cart_abandonment_rate, 1),
    "Email_Open_Rate": np.round(email_open_rate, 1),
    "Customer_Service_Contacts": service_contacts,
    "Satisfaction_Score": np.round(satisfaction_score, 1),
    "NPS": np.round(nps, 0).astype(int),
    "Referrals": referrals,
    "Loyalty_Points": loyalty_points,
    "Complaints": complaints,
    "Favorite_Category": favorite_category,
    "Num_Categories_Shopped": num_categories,
    "Discount_Usage_Rate": np.round(discount_usage_rate, 1),
    "Returns_Rate": np.round(returns_rate, 1),
    "Churn_Risk_Score": np.round(churn_score, 1),
    "Churned": churned,
}

df = pd.DataFrame(data)

# Add some missing values (realistic)
missing_columns = ["Email_Open_Rate", "NPS", "Satisfaction_Score"]
for col in missing_columns:
    missing_indices = np.random.choice(
        df.index, size=int(len(df) * 0.03), replace=False
    )
    df.loc[missing_indices, col] = np.nan

# ============================================================================
# PART 9: Save and Report
# ============================================================================

print()
print("=" * 70)
print("Saving CSV file...")
print("=" * 70)

df.to_csv("customer_data.csv", index=False)
print(f"✓ customer_data.csv saved ({len(df)} customers)")

print()
print("=" * 70)
print("Dataset Summary")
print("=" * 70)
print()
print(f"Total Customers: {len(df)}")
print(f"Features: {len(df.columns) - 2}")  # Excluding ID and Churned
print(f"Target Variables:")
print(f"  • Classification: Churned (binary)")
print(f"  • Clustering: All features (unsupervised)")
print()

print("CHURN DISTRIBUTION:")
churned_count = df["Churned"].sum()
active_count = len(df) - churned_count
print(f"  • Active Customers: {active_count} ({active_count / len(df) * 100:.1f}%)")
print(f"  • Churned Customers: {churned_count} ({churned_count / len(df) * 100:.1f}%)")
print()

print("CUSTOMER SEGMENTS (Natural Groups for Clustering):")
print(f"  • Membership Types: {df['Membership_Type'].value_counts().to_dict()}")
print(f"  • Location Types: {df['Location_Type'].value_counts().to_dict()}")
print(f"  • Avg Purchases by Membership:")
for mtype in ["Basic", "Premium", "VIP"]:
    avg = df[df["Membership_Type"] == mtype]["Total_Purchases"].mean()
    print(f"      - {mtype}: {avg:.1f} purchases")
print()

print("KEY STATISTICS:")
print(f"  • Average Annual Income: ${df['Annual_Income'].mean():,.0f}")
print(f"  • Average Total Spent: ${df['Total_Amount_Spent'].mean():,.2f}")
print(f"  • Average Satisfaction: {df['Satisfaction_Score'].mean():.1f}/10")
print(f"  • Average Purchase Frequency: {df['Purchase_Frequency'].mean():.2f}/month")
print()

print("FEATURE TYPES:")
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features = [
    f for f in numerical_features if f not in ["Customer_ID", "Churned"]
]
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
categorical_features = [f for f in categorical_features if f != "Customer_ID"]

print(f"  • Numerical Features: {len(numerical_features)}")
print(f"  • Categorical Features: {len(categorical_features)}")
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

print("=" * 70)
print("Project Approaches:")
print("=" * 70)
print()
print("CLASSIFICATION APPROACH (Predict Churn):")
print("  • Target: Churned (0 = Active, 1 = Churned)")
print("  • Models to try:")
print("      - Logistic Regression (baseline)")
print("      - Decision Tree Classifier")
print("      - Random Forest Classifier")
print("      - Support Vector Machine (SVM)")
print("  • Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC")
print("  • Business value: Identify at-risk customers for retention")
print()
print("CLUSTERING APPROACH (Customer Segmentation):")
print("  • Remove 'Churned' column (unsupervised)")
print("  • Algorithms to try:")
print("      - K-Means (test k=3,4,5)")
print("      - Hierarchical Clustering")
print("      - DBSCAN (density-based)")
print("  • Evaluation: Silhouette Score, Elbow Method")
print("  • Business value: Understand customer segments for marketing")
print()
print("HYBRID APPROACH (Both!):")
print("  • Use clustering to find segments")
print("  • Analyze churn patterns within each segment")
print("  • Provide segment-specific retention strategies")
print()

print("KEY FEATURES FOR ANALYSIS:")
print("  • Behavioral: Purchase frequency, Days since last purchase")
print("  • Engagement: Email open rate, Visits per month")
print("  • Satisfaction: Satisfaction score, NPS, Complaints")
print("  • Loyalty: Loyalty points, Referrals, Membership type")
print("  • Value: Total amount spent, Avg order value")
print()

print("=" * 70)
print("Data generation complete!")
print("You can now proceed with Project 5: Advanced Analytics")
print("=" * 70)
print()
print("NOTE: Do NOT modify this CSV file before starting your project.")
print("      Choose Classification OR Clustering (or attempt both!)")
print("=" * 70)
