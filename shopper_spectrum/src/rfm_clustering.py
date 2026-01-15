import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import timedelta

def build_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalAmount': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def train_kmeans(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_map = {
        0: "High-Value",
        1: "Regular",
        2: "Occasional",
        3: "At-Risk"
    }

    rfm['Segment'] = rfm['Cluster'].map(cluster_map)

    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    return rfm
