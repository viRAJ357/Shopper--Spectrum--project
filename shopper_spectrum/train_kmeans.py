import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

# -------------------------------------------------
# Resolve paths safely
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "online_retail.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Drop missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Convert InvoiceDate to datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Remove cancelled orders (negative quantity)
df = df[df["Quantity"] > 0]

# -------------------------------------------------
# Create RFM features
# -------------------------------------------------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,  # Recency
    "InvoiceNo": "nunique",                                   # Frequency
    "UnitPrice": lambda x: (x * df.loc[x.index, "Quantity"]).sum()  # Monetary
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# -------------------------------------------------
# Train KMeans
# -------------------------------------------------
X = rfm[["Recency", "Frequency", "Monetary"]]

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# -------------------------------------------------
# Save model
# -------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(kmeans, MODEL_PATH)

print("✅ KMeans model trained and saved successfully")
print(f"📁 Model saved at: {MODEL_PATH}")
