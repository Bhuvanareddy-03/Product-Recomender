import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Clustering Recommender", layout="wide")
st.title("🛍️ Product Recommendation using Clustering")

uploaded_file = st.file_uploader("Upload your ratings CSV file", type=["csv"])
if uploaded_file:
    if uploaded_file.size > 5_000_000:
        st.error("File too large. Please upload a smaller sample (under 5MB).")
        st.stop()

    try:
        df_raw = pd.read_csv(uploaded_file)
        st.subheader("📄 Raw File Preview")
        st.write(df_raw.head(10))

        df = df_raw.copy()
        if df.shape[1] != 4:
            st.error("Uploaded file must have exactly 4 columns: userId, productId, rating, timestamp.")
            st.stop()

        df.columns = ['userId', 'productId', 'rating', 'timestamp']
        df.drop(columns=['timestamp'], inplace=True)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.dropna(inplace=True)

        df_sample = df if len(df) < 10000 else df.sample(n=10000, random_state=42)
        matrix = df_sample.pivot_table(index='userId', columns='productId', values='rating').fillna(0)
        st.write("✅ Matrix shape:", matrix.shape)

        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            st.error("Not enough data to perform clustering.")
            st.stop()

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(matrix)

        # PCA for clustering
        pca = PCA(n_components=min(30, scaled_data.shape[1]), random_state=42)
        reduced_data = pca.fit_transform(scaled_data)

        # PCA for visualization
        pca_vis = PCA(n_components=2, random_state=42)
        vis_data = pca_vis.fit_transform(scaled_data)

        explained = pca.explained_variance_ratio_
        explained_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(explained))],
            'Explained Variance': explained
        })
        st.subheader("🔍 PCA Explained Variance Ratio")
        st.dataframe(explained_df)

        cumulative = np.cumsum(explained)
        st.subheader("📈 Cumulative Explained Variance")
        st.line_chart(cumulative)

        # Clustering
        scores = {}
        try:
            kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=512)
            matrix['Cluster_KMeans'] = kmeans.fit_predict(reduced_data)
            scores['Cluster_KMeans'] = silhouette_score(reduced_data, matrix['Cluster_KMeans'])
        except Exception as e:
            st.warning(f"KMeans failed: {e}")
            scores['Cluster_KMeans'] = "N/A"

        try:
            hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
            matrix['Cluster_HC'] = hc.fit_predict(reduced_data)
            scores['Cluster_HC'] = silhouette_score(reduced_data, matrix['Cluster_HC'])
        except Exception as e:
            st.warning(f"Hierarchical clustering failed: {e}")
            scores['Cluster_HC'] = "N/A"

        try:
            dbscan = DBSCAN(eps=2.5, min_samples=5)
            db_labels = dbscan.fit_predict(reduced_data)
            matrix['Cluster_DBSCAN'] = db_labels
            if len(set(db_labels)) > 1 and -1 not in set(db_labels):
                scores['Cluster_DBSCAN'] = silhouette_score(reduced_data, db_labels)
            else:
                scores['Cluster_DBSCAN'] = "N/A"
        except Exception as e:
            st.warning(f"DBSCAN failed: {e}")
            scores['Cluster_DBSCAN'] = "N/A"

        st.session_state['matrix'] = matrix.copy()

        available_models = [col for col in ['Cluster_KMeans', 'Cluster_HC', 'Cluster_DBSCAN'] if col in matrix.columns]
        st.sidebar.header("🔧 Model Selection")
        model_choice = st.sidebar.selectbox("Choose clustering model", available_models)

        def recommend_products(user_id, cluster_label_col, matrix):
            if cluster_label_col not in matrix.columns:
                return f"Model '{cluster_label_col}' not found."
            if user_id not in matrix.index:
                return "User ID not found."

            user_cluster = matrix.loc[user_id, cluster_label_col]
            if user_cluster == -1:
                return "User is in noise cluster (-1) in DBSCAN. No recommendations available."

            cluster_users = matrix[matrix[cluster_label_col] == user_cluster]
            if cluster_users.shape[0] < 2:
                return f"No similar users in cluster {user_cluster}."

            cluster_users = cluster_users.drop(columns=available_models, errors='ignore')
            cluster_users = cluster_users.apply(pd.to_numeric, errors='coerce')
            cluster_users = cluster_users.dropna(axis=1, how='all')

            if cluster_users.empty:
                return "No product ratings in this cluster."

            mean_ratings = cluster_users.mean().sort_values(ascending=False)
            if mean_ratings.empty:
                return "No ratings found after averaging."

            return mean_ratings.head(5)

        st.subheader("🎯 Get Recommendations")
        if 'matrix' in st.session_state:
            matrix = st.session_state['matrix']
            selected_user = st.selectbox("Select a User ID", matrix.index)
            if st.button("Recommend Products"):
                try:
                    recommendations = recommend_products(selected_user, model_choice, matrix)
                    if isinstance(recommendations, str):
                        st.warning(recommendations)
                    else:
                        st.write(f"Top recommended products for user {selected_user}:")
                        st.dataframe(recommendations)
                except Exception as e:
                    st.error(f"Recommendation error: {e}")
        else:
            st.warning("Matrix not available. Upload a file first.")

        st.subheader("📈 Model Comparison")
        st.dataframe(pd.DataFrame(scores.items(), columns=['Model', 'Silhouette Score']))

        # Visualization
        st.subheader("🖼️ DBSCAN Cluster Visualization")
        if 'Cluster_DBSCAN' in matrix.columns:
            fig, ax = plt.subplots()
            labels = matrix['Cluster_DBSCAN'].values
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap('tab10', len(unique_labels))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                color = 'gray' if label == -1 else colors(i)
                ax.scatter(vis_data[mask, 0], vis_data[mask, 1], label=f'Cluster {label}', alpha=0.6, c=[color])

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("DBSCAN Clusters (PCA 2D)")
            ax.legend()
            st.pyplot(fig)

        st.success("🎉 All steps completed successfully!")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
