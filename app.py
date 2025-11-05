import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="🔗 Product Recommender", layout="centered")
st.title("🔗 Scalable Product Recommendation using Hierarchical Clustering")
st.write("Upload your ratings CSV file")

uploaded_file = st.file_uploader("Upload ratings_short.csv", type="csv")

def load_and_cluster(df):
    # Rename columns to standard format
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    required_cols = {'userid', 'productid', 'rating'}
    if not required_cols.issubset(df.columns):
        st.error("❌ CSV must contain 'userid', 'productid', and 'rating' columns.")
        return None

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df.dropna(subset=['userid', 'productid', 'rating'], inplace=True)

    # Sample for memory efficiency
    df_sample = df.sample(n=min(3000, len(df)), random_state=42)
    matrix = df_sample.pivot_table(index='userid', columns='productid', values='rating').fillna(0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(matrix)

    pca = PCA(n_components=min(30, scaled_data.shape[1]), random_state=42)
    reduced_data = pca.fit_transform(scaled_data)

    hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
    hc_labels = hc.fit_predict(reduced_data)
    matrix['cluster_hc'] = hc_labels

    return matrix

def recommend_products(user_id, matrix):
    if user_id not in matrix.index:
        return ["User ID not found."]
    
    user_cluster = matrix.loc[user_id, 'cluster_hc']
    cluster_users = matrix[matrix['cluster_hc'] == user_cluster]
    cluster_users = cluster_users.drop(columns=['cluster_hc'], errors='ignore')
    cluster_users = cluster_users.apply(pd.to_numeric, errors='coerce')
    cluster_users = cluster_users.dropna(axis=1, how='all')

    if cluster_users.empty:
        return ["No product ratings in this cluster."]

    mean_ratings = cluster_users.mean().sort_values(ascending=False)
    return mean_ratings.head(5).to_dict()

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded: {uploaded_file.name} — {df.shape[0]} rows")
        matrix = load_and_cluster(df)

        if matrix is not None:
            user_ids = matrix.index.tolist()
            selected_user = st.selectbox("Select User ID", user_ids)

            if st.button("Recommend Products"):
                recommendations = recommend_products(selected_user, matrix)
                st.subheader("Top Recommended Products:")
                for product, score in recommendations.items():
                    st.write(f"📦 Product {product} — Avg Rating: {score:.2f}")
    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")

