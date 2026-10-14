import pandas as pd

# Load the dataset
df = pd.read_csv("data/Mall_Customers.csv")

print("========== CUSTOMERIQ: DATA PREPARATION ==========")

# Select features for customer segmentation
features = df[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].copy()

# Check for missing values
print("\nMissing values:")
print(features.isnull().sum())

# Check dataset dimensions
print("\nDataset shape:")
print(features.shape)

# Display the first five records
print("\nSelected customer features:")
print(features.head())

# Validate the data
if features.isnull().values.any():
    print("\nMissing values detected. Please clean the dataset.")
else:
    print("\nData preparation completed successfully!")

# Save the prepared features
features.to_csv("data/prepared_customers.csv", index=False)

print("\nPrepared data saved to data/prepared_customers.csv")