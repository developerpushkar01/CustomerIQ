import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load prepared customer data
df = pd.read_csv("data/prepared_customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate WCSS for different cluster counts
wcss = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    wcss.append(model.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(9, 6))

plt.plot(
    range(1, 11),
    wcss,
    marker="o"
)

plt.title("Elbow Method - CustomerIQ")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Elbow Method completed successfully.")
print("\nWCSS values:")

for k, value in enumerate(wcss, start=1):
    print(f"K = {k}: WCSS = {value:.2f}")