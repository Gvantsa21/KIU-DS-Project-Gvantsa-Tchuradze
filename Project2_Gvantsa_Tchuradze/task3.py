import pandas as pd
import numpy as np

# Load cleaned datasets
customers = pd.read_csv("Project2_Gvantsa_Tchuradze/customers_clean.csv")
products = pd.read_csv("Project2_Gvantsa_Tchuradze/products_clean.csv")
transactions = pd.read_csv("Project2_Gvantsa_Tchuradze/transactions_clean.csv")

# Remove duplicate keys to ensure merge works correctly
customers = customers.drop_duplicates(subset="customer_id")
products = products.drop_duplicates(subset="product_id")

# Merge transactions with customers, then with products
merged_df = (
    transactions
    .merge(customers, on="customer_id", how="inner", validate="many_to_one")
    .merge(products, on="product_id", how="inner", validate="many_to_one")
)

# Check for unmatched references and data loss
unmatched_customers = transactions[~transactions["customer_id"].isin(customers["customer_id"])].shape[0]
unmatched_products = transactions[~transactions["product_id"].isin(products["product_id"])].shape[0]
data_loss = len(transactions) - len(merged_df)

print(f"Unmatched customer IDs: {unmatched_customers}")
print(f"Unmatched product IDs: {unmatched_products}")
print(f"Rows lost during merge: {data_loss}")

# FINANCIAL FEATURES
merged_df["total_amount"] = merged_df["price"] * merged_df["quantity"]
merged_df["discount"] = np.where(merged_df["quantity"] > 3, merged_df["total_amount"] * 0.10, 0)
merged_df["final_amount"] = merged_df["total_amount"] - merged_df["discount"]

# TEMPORAL FEATURES
merged_df["transaction_date"] = pd.to_datetime(merged_df["transaction_date"], errors="coerce")
merged_df["transaction_month"] = merged_df["transaction_date"].dt.month_name()
merged_df["transaction_day_of_week"] = merged_df["transaction_date"].dt.day_name()
merged_df["customer_age_at_purchase"] = merged_df["age"]

# CATEGORICAL FEATURES
# Customer segment based on total spending
spending = merged_df.groupby("customer_id")["final_amount"].sum().reset_index()
spending["customer_segment"] = pd.cut(
    spending["final_amount"],
    bins=[-np.inf, 500, 1000, np.inf],
    labels=["Low", "Medium", "High"]
)
merged_df = merged_df.merge(spending[["customer_id", "customer_segment"]], on="customer_id", how="left")

# Age group
merged_df["age_group"] = pd.cut(
    merged_df["age"],
    bins=[17, 30, 45, 60, np.inf],
    labels=["18-30", "31-45", "46-60", "61+"]
)

# Weekend flag
merged_df["is_weekend"] = merged_df["transaction_day_of_week"].isin(["Saturday", "Sunday"])

# Save enhanced dataset
merged_df.to_csv("enhanced_transactions.csv", index=False)

# ---------------- ADVANCED ANALYSIS ----------------

# 1. Revenue Analysis
rev_by_cat = merged_df.groupby("category")["final_amount"].sum().sort_values(ascending=False)
monthly_rev = (merged_df
               .assign(month=merged_df["transaction_date"].dt.to_period("M"))
               .groupby("month")["final_amount"].sum()
               .sort_index()
               .to_timestamp())
rev_by_country_top5 = merged_df.groupby("country")["final_amount"].sum().sort_values(ascending=False).head(5)
avg_tx_value_by_payment = merged_df.groupby("payment_method")["final_amount"].mean().round(2)

# 2. Customer Behavior
purchases_per_customer = merged_df.groupby(["customer_id","name"])["transaction_id"].count().sort_values(ascending=False).head(10)
avg_spend_by_agegroup = merged_df.groupby("age_group", observed=True)["final_amount"].mean().round(2)
most_pop_cat_by_country = merged_df.groupby(["country","category"])["quantity"].sum().groupby(level=0).idxmax().to_frame(name="country_category_pair")
most_pop_cat_pretty = most_pop_cat_by_country["country_category_pair"].apply(lambda x: pd.Series({"country": x[0], "category": x[1]}))
weekend_summary = merged_df.groupby("is_weekend").agg(transactions=("transaction_id","count"),
                                                      total_revenue=("final_amount","sum"),
                                                      avg_value=("final_amount","mean")).round(2)

# 3. Product Performance
top10_products_revenue = merged_df.groupby(["product_id","product_name"])["final_amount"].sum().sort_values(ascending=False).head(10)
top10_products_quantity = merged_df.groupby(["product_id","product_name"])["quantity"].sum().sort_values(ascending=False).head(10)
avg_tx_by_category = merged_df.groupby("category")["final_amount"].mean().sort_values(ascending=False)
slow_moving_products = merged_df.groupby(["product_id","product_name"])["quantity"].sum().sort_values().head(10)

# 4. Summary Tables
pivot_cat_country = merged_df.pivot_table(index="category", columns="country", values="final_amount", aggfunc="sum", fill_value=0)
crosstab_age_segment = pd.crosstab(merged_df["age_group"], merged_df["customer_segment"])

# Save analysis outputs
rev_by_cat.to_csv("revenue_by_category.csv")
monthly_rev.to_csv("monthly_revenue.csv")
rev_by_country_top5.to_csv("revenue_by_country_top5.csv")
avg_tx_value_by_payment.to_csv("avg_tx_value_by_payment.csv")
purchases_per_customer.to_csv("top10_purchases_per_customer.csv")
avg_spend_by_agegroup.to_csv("avg_spend_by_agegroup.csv")
most_pop_cat_pretty.to_csv("most_pop_category_by_country.csv", index=False)
weekend_summary.to_csv("weekend_vs_weekday.csv")
top10_products_revenue.to_csv("top10_products_by_revenue.csv")
top10_products_quantity.to_csv("top10_products_by_quantity.csv")
avg_tx_by_category.to_csv("category_avg_transaction_value.csv")
slow_moving_products.to_csv("slow_moving_products.csv")
pivot_cat_country.to_csv("pivot_category_country.csv")
crosstab_age_segment.to_csv("crosstab_age_segment.csv")
