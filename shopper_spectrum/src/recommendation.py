import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def build_recommender(df):
    pivot = pd.pivot_table(
        df,
        index='CustomerID',
        columns='Description',
        values='Quantity',
        fill_value=0
    )

    similarity = cosine_similarity(pivot.T)
    similarity_df = pd.DataFrame(
        similarity,
        index=pivot.columns,
        columns=pivot.columns
    )

    joblib.dump(similarity_df, 'models/product_similarity.pkl')
    return similarity_df
