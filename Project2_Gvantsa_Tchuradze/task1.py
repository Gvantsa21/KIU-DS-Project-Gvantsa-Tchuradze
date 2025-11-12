import pandas as pd

# Load CSV files
customers = pd.read_csv("Project2_Gvantsa_Tchuradze/customers.csv")
products = pd.read_csv("Project2_Gvantsa_Tchuradze/products.csv")
transactions = pd.read_csv("Project2_Gvantsa_Tchuradze/transactions.csv")

# Convert date columns to datetime
customers["registration_date"] = pd.to_datetime(customers["registration_date"], errors="coerce")
transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"], errors="coerce")

# Function to explore a DataFrame
def explore_dataframe(df: pd.DataFrame, name: str):
    print(f"\n{'='*30}\nExploring {name}\n{'='*30}\n")
    print("First 5 rows:\n", df.head(), "\n")
    print("Last 5 rows:\n", df.tail(), "\n")
    print(f"Shape: {df.shape}")
    print("Columns and types:\n", df.dtypes, "\n")
    print("Memory usage:")
    print(df.info(memory_usage='deep'))
    print("\nNumerical summary:\n", df.describe(), "\n")
    print("Categorical summary:\n", df.describe(include='object'), "\n")
    print("Unique values per column:\n", df.nunique(), "\n")
    print("Missing values:\n", df.isnull().sum(), "\n")
    print("Duplicate rows:", df.duplicated().sum(), "\n")

# Explore all datasets
explore_dataframe(customers, "Customers")
explore_dataframe(products, "Products")
explore_dataframe(transactions, "Transactions")

# Clean age column
customers["age_clean"] = pd.to_numeric(customers["age"].astype(str).str.extract(r'(\d+)')[0], errors="coerce")

# Standardize country names
country_map = {"US": "United States", "USA": "United States", "U.S.": "United States"}
customers["country_norm"] = customers["country"].replace(country_map).str.strip()

# Customer Analysis
print("\n--- Customer Analysis ---\n")
print("Customers by country:\n", customers["country_norm"].value_counts().sort_values(ascending=False), "\n")
print("Age distribution:\n", customers["age_clean"].agg(["min", "max", "mean", "median"]).round(2), "\n")
regs_by_month = customers.assign(month=customers["registration_date"].dt.to_period("M")) \
                         .groupby("month")["customer_id"].count()
print("Registrations per month:\n", regs_by_month, "\n")
print("Month with most new registrations:", regs_by_month.idxmax(), "\n")

# Product Analysis
print("\n--- Product Analysis ---\n")
print("Products by category:\n", products["category"].value_counts(), "\n")
print("Average price per category:\n", products.groupby("category")["price"].mean().round(2).sort_values(ascending=False), "\n")
print("Out-of-stock products:\n", products[products["stock"].fillna(0) == 0][["product_id", "product_name", "category", "stock"]], "\n")

# Transaction Analysis
print("\n--- Transaction Analysis ---\n")
print("Transactions by payment method:\n", transactions["payment_method"].value_counts(), "\n")

# Most popular product by number of transactions
tx_count = transactions.groupby("product_id")["transaction_id"].count()
most_tx_product_id = tx_count.idxmax()
products_lookup = products.set_index("product_id")["product_name"]
print("Most popular product by transactions:",
      f"{products_lookup.get(most_tx_product_id, most_tx_product_id)} (transactions: {tx_count.max()})")

# Customer with most purchases
cust_tx_count = transactions.groupby("customer_id")["transaction_id"].count()
top_cust_id = cust_tx_count.idxmax()
cust_lookup = customers.set_index("customer_id")[["name", "email"]]
top_cust = cust_lookup.loc[top_cust_id] if top_cust_id in cust_lookup.index else None
print("Top customer by purchases:",
      f"{top_cust['name']} ({top_cust['email']})" if top_cust is not None else top_cust_id,
      f"with {cust_tx_count.max()} transactions\n")
