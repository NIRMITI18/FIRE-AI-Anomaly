import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# APP CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("🧠 FireAI Anomaly Detection Dashboard")

st.markdown("""
This app performs **statistical, ML, or hybrid anomaly detection** on uploaded sales data.  
It also provides **deep EDA (Exploratory Data Analysis)** to understand your data better.
""")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(upload):
        df = pd.read_csv(upload)

        # Detect key columns automatically
        date_candidates = ['date', 'Date', 'day', 'Day', 'ds', 'timestamp', 'time']
        sales_candidates = ['sales', 'Sales', 'number_sold', 'units_sold', 'quantity', 'amount']
        store_candidates = [c for c in df.columns if 'store' in c.lower()]
        product_candidates = [c for c in df.columns if 'product' in c.lower()]

        date_col = next((col for col in df.columns if col in date_candidates), None)
        sales_col = next((col for col in df.columns if col in sales_candidates), None)

        if date_col is None or sales_col is None or not store_candidates or not product_candidates:
            st.error("❌ Missing required columns: 'date', 'sales', 'store', or 'product'. Please fix your CSV.")
            st.stop()

        df = df.rename(columns={
            date_col: 'date',
            sales_col: 'sales',
            store_candidates[0]: 'store_id',
            product_candidates[0]: 'product_id'
        })

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df

    data = load_data(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head())

    # ---------------------------------------------------
    # EDA SECTION
    # ---------------------------------------------------
    st.markdown("---")
    st.header("🔍 Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Basic Info")
        st.write("Shape of dataset:", data.shape)
        st.write("Missing values per column:")
        st.write(data.isnull().sum())

    with col2:
        st.subheader("🧮 Descriptive Statistics")
        st.write(data.describe())

    # Time features
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month_name()

    # ---------------------------------------------------
    # VISUALIZATIONS
    # ---------------------------------------------------
    st.subheader("📈 Yearly Sales Trend")
    yearly_sales = data.groupby('year')['sales'].sum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(yearly_sales['year'], yearly_sales['sales'], marker='o', color='teal')
    ax1.set_title("Total Sales per Year")
    st.pyplot(fig1)

    st.subheader("📆 Monthly Average Sales")
    monthly_sales = data.groupby('month')['sales'].mean().reindex([
        'January','February','March','April','May','June',
        'July','August','September','October','November','December'
    ])
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    monthly_sales.plot(kind='bar', ax=ax2, color='purple')
    ax2.set_title("Average Monthly Sales")
    st.pyplot(fig2)

    st.subheader("🏪 Store vs Sales Distribution")
    store_sales = data.groupby('store_id')['sales'].sum().reset_index().sort_values('sales', ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=store_sales, x='store_id', y='sales', ax=ax3, palette='Blues')
    ax3.set_title("Total Sales per Store")
    st.pyplot(fig3)

    st.subheader("📦 Product vs Sales Distribution")
    product_sales = data.groupby('product_id')['sales'].sum().reset_index().sort_values('sales', ascending=False)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=product_sales, x='product_id', y='sales', ax=ax4, palette='Greens')
    ax4.set_title("Total Sales per Product")
    st.pyplot(fig4)

    # ---------------------------------------------------
    # CORRELATION ANALYSIS
    # ---------------------------------------------------
    st.subheader("🔗 Correlation Heatmap")
    numeric_data = data.select_dtypes(include=np.number)
    if not numeric_data.empty:
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", ax=ax5)
        st.pyplot(fig5)
    else:
        st.info("No numeric columns available for correlation analysis.")

    # ---------------------------------------------------
    # ANOMALY DETECTION
    # ---------------------------------------------------
    st.markdown("---")
    st.header("⚙️ Anomaly Detection")

    approach = st.radio(
        "Select the approach:",
        ["Statistical", "Machine Learning (Isolation Forest)", "Hybrid"],
        key="approach_radio"
    )

    store_ids = sorted(data['store_id'].unique())
    product_ids = sorted(data['product_id'].unique())

    selected_store = st.selectbox("🏪 Select Store ID", store_ids, key="store_select")
    selected_product = st.selectbox("📦 Select Product ID", product_ids, key="product_select")

    filtered_data = data[(data['store_id'] == selected_store) & (data['product_id'] == selected_product)]

    if filtered_data.empty:
        st.warning("⚠️ No data found for this selection.")
    else:
        st.subheader(f"📈 Sales Trend for Store {selected_store}, Product {selected_product}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(filtered_data['date'], filtered_data['sales'], label='Sales', color='blue')
        ax.set_title('Sales Over Time')
        st.pyplot(fig)

        # Run anomaly detection
        st.subheader("🤖 Running Anomaly Detection...")
        results = filtered_data.copy().sort_values("date")

        if approach in ["Statistical", "Hybrid"]:
            window = 30
            rolling_mean = results['sales'].rolling(window, min_periods=5).mean()
            rolling_std = results['sales'].rolling(window, min_periods=5).std()
            results['z_score'] = (results['sales'] - rolling_mean) / (rolling_std + 1e-9)
            results['stat_anomaly'] = (abs(results['z_score']) > 3).astype(int)

        if approach in ["Machine Learning (Isolation Forest)", "Hybrid"]:
            scaler = StandardScaler()
            scaled_sales = scaler.fit_transform(results[['sales']])
            iso = IsolationForest(contamination=0.05, random_state=42)
            results['ml_anomaly'] = iso.fit_predict(scaled_sales)
            results['ml_anomaly'] = results['ml_anomaly'].map({1: 0, -1: 1})

        if approach == "Hybrid":
            results['anomaly'] = ((results['stat_anomaly'] == 1) | (results['ml_anomaly'] == 1)).astype(int)
        elif approach == "Statistical":
            results['anomaly'] = results['stat_anomaly']
        else:
            results['anomaly'] = results['ml_anomaly']

        anomalies = results[results['anomaly'] == 1]
        st.write(f"🚨 **Detected {len(anomalies)} anomalies using {approach} approach**")

        # Visualize anomalies
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(results['date'], results['sales'], color='blue', label='Normal')
        ax2.scatter(anomalies['date'], anomalies['sales'], color='red', label='Anomaly')
        ax2.set_title(f'Anomaly Detection Results ({approach})')
        ax2.legend()
        st.pyplot(fig2)

        # Download results
        st.subheader("📥 Download Results")
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download CSV with Anomalies",
            csv,
            f"anomaly_results_{approach.replace(' ', '_').lower()}.csv",
            "text/csv",
        )

    st.markdown("---")
    st.markdown("✅ **Developed as part of FireAI Engineering Assessment – Anomaly Detection**")

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
