# Shopper--Spectrum--project
This repo relects the lamentix project work
 Shopper Spectrum – Intelligent Customer Segmentation System
 Project Overview
Shopper Spectrum is a Machine Learning–based customer segmentation system built using RFM Analysis and K-Means Clustering.
The project helps businesses identify valuable customer groups such as Loyal, At-Risk, and High-Value customers, enabling data-driven marketing decisions.

This project was developed as part of the Labmentix Data Science / AIML Internship (Jan Batch).

 Key Objectives
Automate customer segmentation from transactional data

Apply Recency, Frequency, Monetary (RFM) analysis

Use K-Means clustering for unsupervised learning

Provide interactive visual insights via Streamlit dashboard

 Technology Stack
Programming Language: Python 3.8+

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib

Web Framework: Streamlit

IDE: VS Code / Jupyter Notebook

 Project Structure
Shopper-Spectrum-Project/
│
├── shopper_spectrum/
│   ├── data/
│   │   └── online_retail.csv
│   ├── models/
│   │   └── kmeans_model.pkl
│   ├── train_kmeans.py
│   └── app.py
│
├── requirements.txt
└── README.md
 How It Works
Upload retail transaction data (online_retail.csv)

Clean and preprocess data

Generate RFM values per customer

Apply K-Means clustering (k = 4)

Visualize customer segments using Streamlit

 How to Run the Project
pip install -r requirements.txt
python train_kmeans.py
streamlit run app.py
 Output
Customer clusters visualization

Segment distribution bar chart

Customer-level RFM summary table

Future Enhancements
Automatic K selection using Elbow Method

3D RFM visualization using Plotly

Cloud deployment (AWS / Streamlit Cloud)

Real-time database or API integration

 Author Details
Name: Nikhil Kumar
Batch: Jan DS / AIML
Submitted To: Labmentix
Mentor: Mr. Vipul Sonawane
Date: 24 January 2026


