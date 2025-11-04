import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

st.title("🛍️ Product Recommendation using Clustering")

uploaded_file = st.file_uploader("Upload your ratings CSV file", type=["csv"])
if uploaded_file:
    try:
        # Load and clean data
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[1] != 4:
            st.error("Expected 4 columns: userId, productId, rating, timestamp.")
            st.stop()
        df.columns = ['userId', 'productId', 'rating', 'timestamp']
        df.drop(columns=['timestamp'], inplace=True)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.dropna(inplace=True)

        # Sample and pivot
        df_sample = df.sample(n=min(10000, len(df)), random_state=42)
        user_product_matrix = df_sample.pivot_table(index='userId',
                                                    columns='productId',
                                                    values='rating').fillna(0)

        if user_product_matrix.shape[0] < 2 or user_product_matrix.shape[1] < 2:
            st.error("Not enough data to perform clustering.")
            st.stop()

        # Standardize and reduce dimensions
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(user_product_matrix)
        n_components = min(30, scaled_data.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        reduced_data = pca.fit_transform(scaled_data)

        # Clustering
        kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=512)
        kmeans_labels = kmeans.fit_predict(reduced_data)
        user_product_matrix['Cluster_KMeans'] = kmeans_labels
        kmeans_score = silhouette_score(reduced_data, kmeans_labels)

        hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
        hc_labels = hc.fit_predict(reduced_data)
        user_product_matrix['Cluster_HC'] = hc_labels
        hc_score = silhouette_score(reduced_data, hc_labels)

        best_eps = None
        best_score = -1
        best_labels = None
        for eps in [0.3, 0.5, 1, 1.5, 2, 3]:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            db_labels = dbscan.fit_predict(reduced_data)
            n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            if n_clusters > 1:
                mask = db_labels != -1
                if mask.sum() > 1:
                    score = silhouette_score(reduced_data[mask], db_labels[mask])
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_labels = db_labels

        if best_labels is not None:
            user_product_matrix['Cluster_DBSCAN'] = best_labels
        else:
            best_score = "N/A"

        # Best model button
        st.subheader("📌 Find Best Model")
        if 'best_model' not in st.session_state:
            st.session_state['best_model'] = 'Cluster_KMeans'

        if st.button("Find Best Model"):
            scores = {
                'Cluster_KMeans': kmeans_score,
                'Cluster_HC': hc_score,
                'Cluster_DBSCAN': best_score if best_labels is not None else -1
            }
            best_model = max(scores, key=lambda k: scores[k] if isinstance(scores[k], float) else -1)
            st.session_state['best_model'] = best_model
            st.success(f"Best model: {best_model} (Score: {scores[best_model]:.3f})")

        # Model comparison
        model_scores = {
            'Cluster_KMeans': kmeans_score,
            'Cluster_HC': hc_score,
            'Cluster_DBSCAN': best_score if best_labels is not None else "N/A"
        }
        st.subheader("📈 Model Comparison")
        st.dataframe(pd.DataFrame(model_scores.items(), columns=['Model', 'Silhouette Score']))

        # Sidebar model selection
        st.sidebar.header("🔧 Model Selection")
        model_choice = st.sidebar.selectbox(
            "Choose clustering model",
            ['Cluster_KMeans', 'Cluster_HC', 'Cluster_DBSCAN'],
            index=['Cluster_KMeans', 'Cluster_HC', 'Cluster_DBSCAN'].index(st.session_state['best_model'])
        )

        # Recommendation logic
        def recommend_products(user_id, cluster_label_col):
            if cluster_label_col not in user_product_matrix.columns:
                return f"Selected model '{cluster_label_col}' did not produce valid clusters."
            if user_id not in user_product_matrix.index:
                return "User ID not found in the data sample."
            user_cluster = user_product_matrix.loc[user_id, cluster_label_col]
            cluster_users = user_product_matrix[user_product_matrix[cluster_label_col] == user_cluster]
            if cluster_users.shape[0] < 2:
                return f"No similar users found in cluster {user_cluster}."
            cluster_users = cluster_users.drop(columns=['Cluster_KMeans', 'Cluster_HC', 'Cluster_DBSCAN'], errors='ignore')
            mean_ratings = cluster_users.mean().sort_values(ascending=False)
            if mean_ratings.empty:
                return "No product ratings available in this cluster."
            return mean_ratings.head(5)

        # Recommendation section
        st.subheader("🎯 Get Recommendations")
        selected_user = st.selectbox("Select a User ID", user_product_matrix.index)
        if st.button("Recommend Products"):
            try:
                recommendations = recommend_products(selected_user, cluster_label_col=model_choice)
                if isinstance(recommendations, str):
                    st.warning(recommendations)
                else:
                    st.write(f"Top recommended products for user {selected_user}:")
                    st.dataframe(recommendations)
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")

    except Exception as e:
        st.error(f"App failed to run: {e}")
