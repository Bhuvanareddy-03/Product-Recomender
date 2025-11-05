import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Scalable HC Recommender", layout="wide")
st.title("🔗 Scalable Product Recommendation using Hierarchical Clustering")

uploaded_file = st.file_uploader("Upload your ratings CSV file", type=["csv"])
if uploaded_file:
    try:
        # Read in chunks and keep only required columns
        chunks = pd.read_csv(uploaded_file, chunksize=100000)
        df_list = []
        for chunk in chunks:
            chunk.columns = [col.strip().lower() for col in chunk.columns]
            if {'userid', 'productid', 'rating', 'date'}.issubset(set(chunk.columns)):
                df_list.append(chunk[['userid', 'productid', 'rating', 'date']])
        if not df_list:
            st.error("No valid data found in file.")
            st.stop()

        df = pd.concat(df_list, ignore_index=True)

        # Clean and sample
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.dropna(inplace=True)
        if len(df) > 20000:
            df = df.sample(n=20000, random_state=42)

        matrix = df.pivot_table(index='userid', columns='productid', values='rating').fillna(0)
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            st.error("Not enough data for clustering.")
            st.stop()

        # Scale and reduce
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(matrix)

        pca = PCA(n_components=min(30, scaled_data.shape[1]), random_state=42)
        reduced_data = pca.fit_transform(scaled_data)

        pca_vis = PCA(n_components=2, random_state=42)
        vis_data = pca_vis.fit_transform(scaled_data)

        # Hierarchical Clustering
        hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
        hc_labels = hc.fit_predict(reduced_data)
        matrix['cluster_hc'] = hc_labels

        # Silhouette Score
        try:
            hc_score = silhouette_score(reduced_data, hc_labels)
        except:
            hc_score = "N/A"

        # Visualization
        st.subheader("🖼️ Cluster Visualization")
        fig, ax = plt.subplots()
        unique_labels = np.unique(hc_labels)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))

        for i, label in enumerate(unique_labels):
            mask = hc_labels == label
            ax.scatter(vis_data[mask, 0], vis_data[mask, 1], label=f'Cluster {label}', alpha=0.6, c=[colors(i)])

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Hierarchical Clusters (PCA 2D)")
        ax.legend()
        st.pyplot(fig)

        # Score display
        st.subheader("📈 Silhouette Score")
        st.write(f"Hierarchical Clustering Silhouette Score: **{hc_score}**")

        # Recommendation logic
        def recommend_products(user_id, matrix):
            if 'cluster_hc' not in matrix.columns:
                return "Model not found."
            if user_id not in matrix.index:
                return "User ID not found."

            user_cluster = matrix.loc[user_id, 'cluster_hc']
            cluster_users = matrix[matrix['cluster_hc'] == user_cluster]

            if cluster_users.shape[0] < 2:
                return f"No similar users in cluster {user_cluster}."

            cluster_users = cluster_users.drop(columns=['cluster_hc'], errors='ignore')
            cluster_users = cluster_users.apply(pd.to_numeric, errors='coerce')
            cluster_users = cluster_users.dropna(axis=1, how='all')

            if cluster_users.empty:
                return "No product ratings in this cluster."

            mean_ratings = cluster_users.mean().sort_values(ascending=False)
            if mean_ratings.empty:
                return "No ratings found after averaging."

            return mean_ratings.head(5)

        # Recommendation section
        st.subheader("🎯 Get Recommendations")
        selected_user = st.selectbox("Select a User ID", matrix.index)
        if st.button("Recommend Products"):
            recommendations = recommend_products(selected_user, matrix)
            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                st.write(f"Top recommended products for user {selected_user}:")
                st.dataframe(recommendations)

        st.success("🎉 App now handles large data efficiently!")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")

