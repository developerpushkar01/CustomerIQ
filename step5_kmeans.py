import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. LOAD CUSTOMER DATA
# ==========================================

df = pd.read_csv("data/prepared_customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# ==========================================
# 2. SCALE THE DATA
# ==========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. TRAIN THE K-MEANS MODEL
# ==========================================

K = 5

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

# Assign a cluster to every customer
df["Segment"] = kmeans.fit_predict(X_scaled)

# ==========================================
# 4. DISPLAY RESULTS
# ==========================================

print("\n========== CUSTOMERIQ SEGMENTATION ==========")

print("\nFirst 10 customers with assigned segments:")
print(df.head(10))

print("\nNumber of customers in each segment:")
print(df["Segment"].value_counts().sort_index())

# ==========================================
# 5. CALCULATE SEGMENT PROFILES
# ==========================================

segment_profiles = df.groupby("Segment").agg(
    Customer_Count=("Segment", "count"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending_Score=("Spending Score (1-100)", "mean")
).round(2)

print("\nCustomer Segment Profiles:")
print(segment_profiles)

# ==========================================
# 6. VISUALIZE CUSTOMER SEGMENTS
# ==========================================

plt.figure(figsize=(10, 7))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Segment"],
    cmap="viridis",
    s=80,
    alpha=0.8
)

# Convert cluster centers back to the original scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250,
    color="red",
    edgecolor="black",
    label="Cluster Centers"
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (thousands of dollars)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

# Save graph
plt.savefig("data/customer_segments.png", dpi=300)

plt.show()

# ==========================================
# 7. SAVE SEGMENTED CUSTOMER DATA
# ==========================================

df.to_csv("data/segmented_customers.csv", index=False)

segment_profiles.to_csv("data/segment_profiles.csv")

print("\nSegmentation completed successfully!")
print("Saved: data/segmented_customers.csv")
print("Saved: data/segment_profiles.csv")
print("Saved: data/customer_segments.png")