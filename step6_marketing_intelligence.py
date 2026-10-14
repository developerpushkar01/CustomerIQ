import pandas as pd

# ==========================================
# 1. LOAD SEGMENT PROFILES
# ==========================================

df = pd.read_csv("data/segmented_customers.csv")

profiles = df.groupby("Segment").agg(
    Customer_Count=("Segment", "count"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).reset_index()

# ==========================================
# 2. CALCULATE DATASET BENCHMARKS
# ==========================================

income_benchmark = df["Annual Income (k$)"].median()
spending_benchmark = df["Spending Score (1-100)"].median()

# ==========================================
# 3. GENERATE MARKETING RECOMMENDATIONS
# ==========================================

def generate_recommendation(row):
    high_income = row["Average_Income"] >= income_benchmark
    high_spending = row["Average_Spending"] >= spending_benchmark

    if high_income and high_spending:
        return (
            "Premium Customers",
            "Offer VIP rewards, exclusive products, "
            "and personalized loyalty benefits."
        )

    elif high_income and not high_spending:
        return (
            "High-Income Potential",
            "Use personalized recommendations, "
            "product demonstrations, and targeted offers."
        )

    elif not high_income and high_spending:
        return (
            "Active Value Seekers",
            "Promote loyalty points, affordable bundles, "
            "and complementary products."
        )

    else:
        return (
            "Budget-Conscious Customers",
            "Test value bundles, introductory offers, "
            "and relevant discounts."
        )

# ==========================================
# 4. APPLY THE INTELLIGENCE ENGINE
# ==========================================

results = profiles.apply(
    generate_recommendation,
    axis=1,
    result_type="expand"
)

profiles["Customer_Type"] = results[0]
profiles["Marketing_Recommendation"] = results[1]

# ==========================================
# 5. DISPLAY RESULTS
# ==========================================

print("\n========== CUSTOMERIQ MARKETING INTELLIGENCE ==========")

print(f"\nMedian annual income: ${income_benchmark:.2f}k")
print(f"Median spending score: {spending_benchmark:.2f}")

for _, row in profiles.iterrows():
    print("\n" + "=" * 55)
    print(f"Segment: {row['Segment']}")
    print(f"Customer type: {row['Customer_Type']}")
    print(f"Number of customers: {row['Customer_Count']}")
    print(f"Average income: ${row['Average_Income']:.2f}k")
    print(f"Average spending score: {row['Average_Spending']:.2f}")
    print(f"Marketing recommendation: {row['Marketing_Recommendation']}")

# ==========================================
# 6. SAVE RESULTS
# ==========================================

profiles.to_csv(
    "data/marketing_intelligence.csv",
    index=False
)

print("\nMarketing intelligence generated successfully!")
print("Saved to: data/marketing_intelligence.csv")