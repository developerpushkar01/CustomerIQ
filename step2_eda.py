
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set graph appearance
sns.set_theme(style="whitegrid")

# Load the dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Display dataset information
print("\n========== DATASET OVERVIEW ==========")

print("\nFirst Five Customers:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# Graph 1: Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=15, kde=True)
plt.title("Distribution of Customer Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Graph 2: Annual Income Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Annual Income (k$)", bins=15, kde=True)
plt.title("Distribution of Annual Income")
plt.xlabel("Annual Income (thousands of dollars)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Graph 3: Spending Score Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Spending Score (1-100)", bins=15, kde=True)
plt.title("Distribution of Customer Spending Scores")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Graph 4: Gender Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Gender")
plt.title("Customer Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Graph 5: Annual Income vs Spending Score
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Gender",
    s=80
)
plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (thousands of dollars)")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.show()

# Graph 6: Correlation Heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.show()

print("\nEDA completed successfully!")

