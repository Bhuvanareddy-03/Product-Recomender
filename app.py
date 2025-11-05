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

        # Sidebar: DBSCAN sliders
        st.sidebar.header("🔧 DBSCAN Parameters")
        eps_val = st.sidebar.slider("DBSCAN eps", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
        min_samples_val = st.sidebar.slider("DBSCAN min_samples", min_value=3, max_value=20, value=5, step=1)

        # Clustering
        scores = {}
        labels_dict = {}

        try:
            kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=512)
            labels_dict['Cluster_KMeans'] = kmeans.fit_predict(reduced_data)
            scores['Cluster_KMeans'] = silhouette_score(reduced_data, labels_dict['Cluster_KMeans'])
        except:
            scores['Cluster_KMeans'] = "N/A"

        try:
            hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
            labels_dict['Cluster_HC'] = hc.fit_predict(reduced_data)
            scores['Cluster_HC'] = silhouette_score(reduced_data, labels_dict['Cluster_HC'])
        except:
            scores['Cluster_HC'] = "N/A"

        try:
            dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
            db_labels = dbscan.fit_predict(reduced_data)
            labels_dict['Cluster_DBSCAN'] = db_labels
            if len(set(db_labels)) > 1 and -1 not in set(db_labels):
                scores['Cluster_DBSCAN'] = silhouette_score(reduced_data, db_labels)
            else:
                scores['Cluster_DBSCAN'] = "N/A"
        except:
            scores['Cluster_DBSCAN'] = "N/A"

        st.session_state['matrix'] = matrix.copy()
        st.session_state['labels_dict'] = labels_dict

        # Sidebar: model selection
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Model Selection")
        available_models = list(labels_dict.keys())
        model_choice = st.sidebar.selectbox("Choose clustering model", available_models)

        # Sidebar: best model button
        if st.sidebar.button("Select Best Model"):
            best_model = max(
                [(m, s) for m, s in scores.items() if isinstance(s, float)],
                key=lambda x: x[1],
                default=(None, None)
            )[0]
            if best_model:
                st.sidebar.success(f"Best model: {best_model}")
                model_choice = best_model
            else:
                st.sidebar.warning("No valid silhouette scores found.")

        # Visualization
        st.subheader("🖼️ Cluster Visualization")
        if model_choice in labels_dict:
            fig, ax = plt.subplots()
            labels = labels_dict[model_choice]
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap('tab10', len(unique_labels))

            for i, label in enumerate(unique_labels):
                mask = labels == label
                color = 'gray' if label == -1 else colors(i)
                ax.scatter(vis_data[mask, 0], vis_data[mask, 1], label=f'Cluster {label}', alpha=0.6, c=[color])

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"{model_choice} Clusters (PCA 2D)")
            ax.legend()
            st.pyplot(fig)

        # Model comparison
        st.subheader("📈 Model Comparison")
        st.dataframe(pd.DataFrame(scores.items(), columns=['Model', 'Silhouette Score']))

        # Recommendation logic
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

        # Recommendation section
        st.subheader("🎯 Get Recommendations")
        selected_user = st.selectbox("Select a User ID", matrix.index)
        if st.button("Recommend Products"):
            matrix[model_choice] = labels_dict[model_choice]
            recommendations = recommend_products(selected_user, model_choice, matrix)
            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                st.write(f"Top recommended products for user {selected_user}:")
                st.dataframe(recommendations)

        st.success("🎉 App is ready with all features!")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
