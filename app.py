import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Page setup
st.set_page_config(page_title="🔗 Product Recommender", layout="centered")
st.title("Product Recommendation using Hierarchical Clustering")
st.write("Upload your ratings CSV file")

# File upload
uploaded_file = st.file_uploader("Upload ratings_short.csv", type="csv")

# Data overview
def show_data_overview(df):
    st.subheader("📊 Data Overview")
     # Show sample with readable timestamp if present
     # Show sample with readable timestamp if present
    if 'time_stamp' in df.columns:
        st.write("**Sample Data with Timestamp:**")
        st.dataframe(df[['userid', 'productid', 'rating', 'time_stamp']].head())
    else:
        st.write("**Sample Data:**")
        st.dataframe(df[['userid', 'productid', 'rating']].head())



    

# Clustering pipeline
def load_and_cluster(df):
    # Normalize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    required_cols = {'userid', 'productid', 'rating'}
    if not required_cols.issubset(df.columns):
        st.error("❌ CSV must contain 'userid', 'productid', and 'rating' columns.")
        return None, None, None

    # Convert rating to numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Convert timestamp if present
    if 'time_stamp' in df.columns:
         df['time_stamp'] = pd.to_datetime(df['time_stamp'], unit='s', errors='coerce').dt.date

   
    df.dropna(subset=['userid', 'productid', 'rating'], inplace=True)

    show_data_overview(df)

    # Sample for memory efficiency
    df_sample = df.sample(n=min(3000, len(df)), random_state=42)
    matrix = df_sample.pivot_table(index='userid', columns='productid', values='rating').fillna(0)

    # Scale and reduce dimensions
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(matrix)

    pca = PCA(n_components=min(30, scaled_data.shape[1]), random_state=42)
    reduced_data = pca.fit_transform(scaled_data)

    # Hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
    hc_labels = hc.fit_predict(reduced_data)
    matrix['cluster_hc'] = hc_labels

    # Silhouette score
    try:
        score = silhouette_score(reduced_data, hc_labels)
    except Exception as e:
        score = None
        st.warning(f"⚠️ Could not compute silhouette score: {e}")

    return matrix, hc_labels, score

# Recommendation logic
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

# Main app logic
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded: {uploaded_file.name} — {df.shape[0]} rows")
        matrix, labels, score = load_and_cluster(df)

        if matrix is not None:
            st.markdown(f"### 📈 Silhouette Score: `{score:.3f}`" if score is not None else "Silhouette score not available.")
            user_ids = matrix.index.tolist()
            selected_user = st.selectbox("Select User ID", user_ids)

            if st.button("Recommend Products"):
                recommendations = recommend_products(selected_user, matrix)
                st.subheader("Top Recommended Products:")
                for product, rating in recommendations.items():
                    st.write(f"📦 Product {product} — Avg Rating: {rating:.2f}")
    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
