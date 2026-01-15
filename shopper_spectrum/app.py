import pandas as pd
import joblib
import os
import streamlit as st

# -------------------------------------------------
# Resolve paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "online_retail.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")

# -------------------------------------------------
# Load model
# -------------------------------------------------
kmeans = joblib.load(MODEL_PATH)

# -------------------------------------------------
# Load & prepare data
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["CustomerID"])
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df[df["Quantity"] > 0]

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "UnitPrice": lambda x: (x * df.loc[x.index, "Quantity"]).sum()
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

rfm["Cluster"] = kmeans.predict(rfm[["Recency", "Frequency", "Monetary"]])

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("🛒 Shopper Spectrum – Customer Segmentation")

st.subheader("Clustered Customers (Sample)")
st.dataframe(rfm.head(20))

st.subheader("Cluster Distribution")
st.bar_chart(rfm["Cluster"].value_counts())
