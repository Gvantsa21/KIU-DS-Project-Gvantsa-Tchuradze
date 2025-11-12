import pandas as pd
import numpy as np

# Load data
customers = pd.read_csv("Project2_Gvantsa_Tchuradze/customers.csv")
products = pd.read_csv("Project2_Gvantsa_Tchuradze/products.csv")
transactions = pd.read_csv("Project2_Gvantsa_Tchuradze/transactions.csv")

# Save original info for report
orig_info = {
    "customers": {"rows": len(customers), "duplicates": customers.duplicated().sum()},
    "products": {"rows": len(products), "duplicates": products.duplicated().sum()},
    "transactions": {"rows": len(transactions), "duplicates": transactions.duplicated(subset=['transaction_id']).sum()}
}

# Data Quality Check
print("=== Customers ===")
print(customers.isnull().sum())
print("Duplicates:", customers.duplicated().sum())
print(customers.dtypes)
print("Countries:", customers['country'].unique())

print("=== Products ===")
print(products.isnull().sum())
print("Negative prices:\n", products[products['price'] < 0])
print("Unrealistic stock:\n", products[(products['stock'] < 0) | (products['stock'] > 500)])
print("Whitespace in product names:\n", products[products['product_name'].str.strip() != products['product_name']])
print("Categories:", products['category'].unique())

print("=== Transactions ===")
print(transactions.isnull().sum())
print("Duplicate transaction IDs:", transactions.duplicated(subset=['transaction_id']).sum())
print("Invalid customer references:", (~transactions['customer_id'].isin(customers['customer_id'])).sum())
print("Future transaction dates:", (pd.to_datetime(transactions['transaction_date'], errors='coerce') > pd.Timestamp.today()).sum())
print("Payment methods:", transactions['payment_method'].unique())


# Cleaning

# 1. Handle missing values
customers = customers.dropna(subset=['email'])
products['price'] = products.groupby('category')['price'].transform(lambda x: x.fillna(x.median()))
transactions['quantity'] = transactions['quantity'].fillna(transactions['quantity'].mode()[0])

# 2. Remove duplicates
customers = customers.drop_duplicates()
products = products.drop_duplicates()
transactions = transactions.drop_duplicates(subset=['transaction_id'])

# 3. Fix data types
type_corrections = {"customers_age": 0, "products_price": 0, "products_stock":0, "transactions_quantity":0}

# Customers age
customers['age'] = pd.to_numeric(customers['age'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
type_corrections["customers_age"] = customers['age'].isnull().sum()

# Dates
customers['registration_date'] = pd.to_datetime(customers['registration_date'], errors='coerce')
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'], errors='coerce')

# Products numeric
products['price'] = pd.to_numeric(products['price'], errors='coerce')
products['stock'] = pd.to_numeric(products['stock'], errors='coerce')
transactions['quantity'] = pd.to_numeric(transactions['quantity'], errors='coerce')
type_corrections["products_price"] = products['price'].isnull().sum()
type_corrections["products_stock"] = products['stock'].isnull().sum()
type_corrections["transactions_quantity"] = transactions['quantity'].isnull().sum()

# 4. Standardize text
customers['country'] = customers['country'].replace({'US':'United States','USA':'United States','U.S.':'United States'})
customers['name'] = customers['name'].str.strip()
customers['email'] = customers['email'].str.strip().str.lower()
products['product_name'] = products['product_name'].str.strip()
products['category'] = products['category'].str.strip().str.lower().str.title()
transactions['payment_method'] = transactions['payment_method'].str.strip().str.title()

# 5. Handle outliers / invalid data
outliers_handled = {"negative_prices": 0, "stock_capped": 0}

# Negative prices
outliers_handled["negative_prices"] = (products['price'] < 0).sum()
products.loc[products['price'] < 0, 'price'] = products['price'].abs()

# Stock > 500
outliers_handled["stock_capped"] = (products['stock'] > 500).sum()
products.loc[products['stock'] > 500, 'stock'] = 500

# Invalid customer references
transactions = transactions[transactions['customer_id'].isin(customers['customer_id'])]

# Future dates
transactions = transactions[transactions['transaction_date'] <= pd.Timestamp.today()]

# Validation & Cleaning Report
report = pd.DataFrame({
    'Dataset': ['customers','products','transactions'],
    'Original Rows':[orig_info['customers']['rows'], orig_info['products']['rows'], orig_info['transactions']['rows']],
    'Cleaned Rows':[len(customers), len(products), len(transactions)],
    'Missing Values After':[customers.isnull().sum().sum(), products.isnull().sum().sum(), transactions.isnull().sum().sum()],
    'Duplicates Removed':[orig_info['customers']['duplicates'] - customers.duplicated().sum(),
                          orig_info['products']['duplicates'] - products.duplicated().sum(),
                          orig_info['transactions']['duplicates'] - transactions.duplicated(subset=['transaction_id']).sum()],
    'Outliers Handled':[outliers_handled["negative_prices"] + outliers_handled["stock_capped"], "-", "-"],
    'Type Corrections':[type_corrections["customers_age"] + type_corrections["products_price"] + type_corrections["products_stock"],
                        "-", type_corrections["transactions_quantity"]]
})
print(report)

# Quick sanity checks
print(customers.head(2))
print(products.head(2))
print(transactions.head(2))

print("Customer ages:", customers['age'].min(), "-", customers['age'].max())
print("Product prices:", products['price'].min(), "-", products['price'].max())
print("Product stock:", products['stock'].min(), "-", products['stock'].max())
print("Transaction quantity:", transactions['quantity'].min(), "-", transactions['quantity'].max())

# Export cleaned datasets
customers.to_csv("customers_clean.csv", index=False)
products.to_csv("products_clean.csv", index=False)
transactions.to_csv("transactions_clean.csv", index=False)
print("Cleaned files saved.")
