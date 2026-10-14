import pandas as pd

# Load the dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Display the first five records
print("\n--- First Five Customers ---")
print(df.head())

# Display dataset dimensions
print("\n--- Dataset Shape ---")
print(df.shape)

# Display column names
print("\n--- Column Names ---")
print(df.columns.tolist())

# Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Display data types
print("\n--- Dataset Information ---")
df.info()