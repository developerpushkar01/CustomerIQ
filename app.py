import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CustomerIQ | Marketing Intelligence",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# CUSTOM STYLING
# ==========================================

# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
    /* Main application */
    .main {
        background-color: #f7f9fc;
    }

    /* Header banner */
    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, #14213d, #2457a7);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white !important;
        font-size: 38px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e4ecff !important;
        font-size: 16px;
    }

    /* Metric cards: dark text on white backgrounds */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #d9e0eb;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label *,
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricDelta"] {
        color: #14213d !important;
        -webkit-text-fill-color: #14213d !important;
    }

    /* Segmentation result containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #d9e0eb;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border-color: #d9e0eb;
    }

    /* Recommendation cards */
    .recommendation {
        padding: 16px;
        background-color: #ffffff;
        color: #14213d;
        border-radius: 12px;
        border-left: 5px solid #2457a7;
        margin-bottom: 12px;
    }

    .recommendation * {
        color: #14213d !important;
    }
</style>
""", unsafe_allow_html=True)
# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>CustomerIQ</h1>
    <p>
        AI-Powered Customer Segmentation
        and Marketing Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Discover customer groups, understand spending patterns, "
    "and generate data-driven marketing recommendations."
)

# ==========================================
# SIDEBAR: DATA SOURCE
# ==========================================

st.sidebar.title("⚙️ Analysis Settings")

data_source = st.sidebar.radio(
    "Choose your data source",
    ["Try Demo Dataset", "Upload Your Own CSV"]
)

uploaded_file = None

if data_source == "Upload Your Own CSV":

    uploaded_file = st.sidebar.file_uploader(
        "Upload customer data",
        type=["csv"]
    )

# ==========================================
# SAMPLE DATASET
# ==========================================

def create_demo_dataset():

    np.random.seed(42)

    customer_count = 300

    customer_ids = np.arange(1, customer_count + 1)

    ages = np.random.randint(
        18, 71, customer_count
    )

    genders = np.random.choice(
        ["Male", "Female"],
        customer_count
    )

    annual_income = np.random.randint(
        15, 140, customer_count
    )

    spending_score = np.random.randint(
        1, 101, customer_count
    )

    demo_df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": genders,
        "Age": ages,
        "Annual Income (k$)": annual_income,
        "Spending Score (1-100)": spending_score
    })

    return demo_df

# ==========================================
# LOAD SELECTED DATA SOURCE
# ==========================================

if data_source == "Try Demo Dataset":

    df = create_demo_dataset()

    st.info(
        "🧪 Demo Mode is active. "
        "You're exploring synthetic sample customer data."
    )

elif uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")
        st.stop()

else:

    st.info(
        "Upload a customer CSV file using the sidebar "
        "or select Try Demo Dataset to explore CustomerIQ."
    )

    st.subheader("What CustomerIQ offers")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📂")
        st.markdown("**Data Analysis**")
        st.write(
            "Explore customer records and numerical patterns."
        )

    with col2:
        st.markdown("### 🤖")
        st.markdown("**AI Segmentation**")
        st.write(
            "Use K-Means to discover groups of similar customers."
        )

    with col3:
        st.markdown("### 🎯")
        st.markdown("**Marketing Intelligence**")
        st.write(
            "Generate segment-level marketing recommendations."
        )

    st.stop()

# ==========================================
# VALIDATE DATA
# ==========================================

if df.empty:
    st.error("The dataset contains no records.")
    st.stop()


if df.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

# Clean column names
df.columns = df.columns.astype(str).str.strip()

# Remove duplicate column names
if df.columns.duplicated().any():
    st.error("The CSV contains duplicate column names. Please fix them.")
    st.stop()

# ==========================================
# DATA PREVIEW
# ==========================================

st.header("1. Customer Data Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    st.metric(
        "Missing Cells",
        f"{int(df.isna().sum().sum()):,}"
    )

with st.expander("Preview uploaded dataset", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)

with st.expander("Dataset statistics"):
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

# ==========================================
# NUMERICAL FEATURES
# ==========================================

st.header("2. Configure Customer Segmentation")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_columns) < 2:
    st.error(
        "Your dataset needs at least two numerical columns "
        "for this version of CustomerIQ."
    )
    st.stop()

# Identify suitable default columns
income_candidates = [
    "Annual Income (k$)",
    "Annual Income",
    "Income",
    "Income ($)",
    "Monthly Income"
]

spending_candidates = [
    "Spending Score (1-100)",
    "Spending Score",
    "Purchase Frequency",
    "Total Spent",
    "Total Spend"
]

default_x = next(
    (c for c in income_candidates if c in numeric_columns),
    numeric_columns[0]
)

default_y = next(
    (c for c in spending_candidates
     if c in numeric_columns and c != default_x),
    next(c for c in numeric_columns if c != default_x)
)

col1, col2 = st.columns(2)

with col1:
    feature_x = st.selectbox(
        "First customer feature",
        numeric_columns,
        index=numeric_columns.index(default_x)
    )

with col2:
    available_y = [
        c for c in numeric_columns
        if c != feature_x
    ]

    if default_y not in available_y:
        default_y = available_y[0]

    feature_y = st.selectbox(
        "Second customer feature",
        available_y,
        index=available_y.index(default_y)
    )

# Prepare selected features
X = df[[feature_x, feature_y]].copy()

# Convert infinite values to missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Convert missing values to the median of each feature
X = X.fillna(X.median())

if X.isna().values.any():
    st.error(
        "The selected features contain no usable numerical values "
        "for at least one column."
    )
    st.stop()

if len(X) < 3:
    st.error("Upload at least three customer records.")
    st.stop()

# Remove constant features
if X.nunique().min() < 2:
    st.error(
        "Both selected features must contain varying values. "
        "Choose different numerical columns."
    )
    st.stop()

# ==========================================
# CLUSTER SETTINGS
# ==========================================


# ==========================================
# AUTOMATIC CLUSTER SELECTION
# ==========================================

st.sidebar.subheader("Segmentation Settings")

max_k = min(10, len(X) - 1)

selection_mode = st.sidebar.radio(
    "Choose segmentation mode",
    ["Automatic", "Manual"]
)

if selection_mode == "Automatic":

    scaler_preview = StandardScaler()
    X_preview = scaler_preview.fit_transform(X)

    wcss = []

    for cluster_count in range(1, max_k + 1):
        model_preview = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10
        )

        model_preview.fit(X_preview)
        wcss.append(model_preview.inertia_)

    points = np.column_stack((
        np.arange(1, max_k + 1, dtype=float),
        np.array(wcss, dtype=float)
    ))

    start = points[0]
    end = points[-1]

    line = end - start
    line_length = np.linalg.norm(line)

    if line_length > 0:
        differences = points - start

        distances = np.abs(
            line[0] * differences[:, 1]
            - line[1] * differences[:, 0]
        ) / line_length

        k = int(np.argmax(distances[1:]) + 2)

    else:
        k = min(5, max_k)

    st.sidebar.success(
        f"Suggested number of segments: {k}"
    )

    with st.sidebar.expander("View Elbow Method"):
        fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))

        ax_elbow.plot(
            range(1, max_k + 1),
            wcss,
            marker="o"
        )

        ax_elbow.axvline(
            k,
            linestyle="--",
            label=f"Suggested K = {k}"
        )

        ax_elbow.set_xlabel("Number of Clusters")
        ax_elbow.set_ylabel("WCSS")
        ax_elbow.set_title("Elbow Method")
        ax_elbow.legend()

        st.pyplot(fig_elbow)
        plt.close(fig_elbow)

else:

    k = st.sidebar.slider(
        "Number of customer segments",
        min_value=2,
        max_value=max_k,
        value=min(5, max_k)
    )
# ==========================================
# RUN K-MEANS
# ==========================================

if st.button("🚀 Analyze Customers", type="primary"):

    with st.spinner("Analyzing customer segments..."):

        # Standardize selected features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train K-Means
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        segment_labels = model.fit_predict(X_scaled)
        # Evaluate clustering quality
        if 2 <= len(set(segment_labels)) < len(X_scaled):
            score = silhouette_score(X_scaled, segment_labels)
        else:
            score = None


        # Preserve original customer records
        results = df.copy()
        results["Segment"] = segment_labels

        # Add processed values for missing selected features
        results[feature_x] = X[feature_x].values
        results[feature_y] = X[feature_y].values

        # ==================================
        # SEGMENT PROFILES
        # ==================================

        profiles = results.groupby("Segment").agg(
            Customer_Count=("Segment", "size"),
            Average_Feature_X=(feature_x, "mean"),
            Average_Feature_Y=(feature_y, "mean")
        ).reset_index()

        profiles["Customer_Percentage"] = (
            profiles["Customer_Count"] / len(results) * 100
        ).round(2)

        # Dataset medians for interpretation
        median_x = results[feature_x].median()
        median_y = results[feature_y].median()

        # ==================================
        # CUSTOMER CLASSIFICATION
        # ==================================

        def classify_segment(row):
            high_x = row["Average_Feature_X"] >= median_x
            high_y = row["Average_Feature_Y"] >= median_y

            if high_x and high_y:
                return (
                    "Premium Customers",
                    "💎",
                    "Focus on loyalty rewards, premium products, "
                    "exclusive offers, and personalized experiences."
                )

            elif high_x and not high_y:
                return (
                    "High-Potential Customers",
                    "🎯",
                    "Test personalized recommendations, targeted "
                    "promotions, and product discovery campaigns."
                )

            elif not high_x and high_y:
                return (
                    "Active Value Seekers",
                    "🛍️",
                    "Promote value bundles, loyalty points, "
                    "and complementary products."
                )

            else:
                return (
                    "Budget-Conscious Customers",
                    "💰",
                    "Test affordable bundles, relevant discounts, "
                    "and introductory offers."
                )

        # Apply classification
        classifications = profiles.apply(
            classify_segment,
            axis=1,
            result_type="expand"
        )

        profiles["Customer_Type"] = classifications[0]
        profiles["Segment_Icon"] = classifications[1]
        profiles["Marketing_Recommendation"] = classifications[2]

        # ==================================
        # STORE RESULTS
        # ==================================

        st.session_state["customeriq_results"] = results
        st.session_state["customeriq_profiles"] = profiles

        st.session_state["customeriq_features"] = (
            feature_x,
            feature_y
        )

        st.session_state["customeriq_k"] = k

        st.session_state["customeriq_centers"] = (
            scaler.inverse_transform(model.cluster_centers_)
        )

        st.session_state["customeriq_silhouette_score"] = score
# ==========================================
# DISPLAY ANALYSIS
# ==========================================

if "customeriq_results" in st.session_state:

    results = st.session_state["customeriq_results"]
    profiles = st.session_state["customeriq_profiles"]

    feature_x, feature_y = st.session_state["customeriq_features"]
    centers = st.session_state["customeriq_centers"]

    st.header("3. Customer Segmentation Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Customers Analyzed", f"{len(results):,}")

    with col2:
        st.metric(
            "Segments Identified",
            st.session_state["customeriq_k"]
        )

    with col3:
        st.metric(
            "Largest Segment",
            f"{profiles['Customer_Count'].max():,} customers"
        )

    silhouette = st.session_state.get("customeriq_silhouette_score")

    with col4:
        st.metric(
            "Silhouette Score",
            f"{silhouette:.3f}" if silhouette is not None else "N/A",
            help="Measures how well-separated the customer segments are. Higher values generally indicate better-separated clusters."
        )

    # ==================================
    # VISUALIZATION
    # ==================================

    st.subheader("Customer Segment Visualization")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=results,
        x=feature_x,
        y=feature_y,
        hue="Segment",
        palette="viridis",
        s=80,
        ax=ax
    )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=250,
        color="red",
        edgecolor="black",
        label="Cluster Centers"
    )

    ax.set_title("Customer Segmentation Using K-Means")
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ==================================
    # SEGMENT PROFILES
    # ==================================

    st.subheader("Customer Segment Profiles")

    st.dataframe(
        profiles,
        use_container_width=True,
        hide_index=True
    )

    # ==================================
    # MARKETING RECOMMENDATIONS
    # ==================================

    st.header("4. Marketing Intelligence")


    st.caption(
        "Customer groups and recommendations are based on "
        "the selected features and their dataset medians."
    )

    # ==================================
    # MARKETING RECOMMENDATIONS
    # ==================================

    for _, row in profiles.iterrows():

        with st.container(border=True):

            st.subheader(
                f"{row['Segment_Icon']} {row['Customer_Type']}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Customers",
                    int(row["Customer_Count"])
                )

            with col2:
                st.metric(
                    "Customer Share",
                    f"{row['Customer_Percentage']:.1f}%"
                )

            with col3:
                st.metric(
                    f"Average {feature_x}",
                    f"{row['Average_Feature_X']:.2f}"
                )

            st.write(
                "**Marketing Recommendation:**",
                row["Marketing_Recommendation"]
            )

    # ==================================
    # DOWNLOAD RESULTS
    # ==================================

    st.header("5. Download Your Results")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="⬇️ Download Segmented Customers",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="customeriq_segmented_customers.csv",
            mime="text/csv"
        )

    with col2:
        st.download_button(
            label="⬇️ Download Marketing Report",
            data=profiles.to_csv(index=False).encode("utf-8"),
            file_name="customeriq_marketing_report.csv",
            mime="text/csv"
        )

    st.success(
        "Analysis complete! Your results are ready to download."
    )

st.divider()

st.caption(
"CustomerIQ | AI-Powered Customer Segmentation "
"and Marketing Intelligence"
)
