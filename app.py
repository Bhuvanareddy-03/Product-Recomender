import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Clustering Recommender", layout="wide")
st.title("🛍️ Product Recommendation using Clustering")

# Step 1: Upload and preview
uploaded_file = st.file_uploader("Upload your ratings CSV file", type=["csv"])
if uploaded_file:
    if uploaded_file.size > 5_000_000:
        st.error("File too large. Please upload a smaller sample (under 5MB).")
        st.stop()

    try:
        df_raw = pd.read_csv(uploaded_file)
        st.subheader("📄 Raw File Preview")
        st.write(df_raw.head(10))

        # Step 2: Clean and validate
        df = df_raw.copy()
        if df.shape[1] != 4:
            st.error("Uploaded file must have exactly 4 columns: userId, productId, rating, timestamp.")
            st.stop()

        df.columns = ['userId', 'productId', 'rating', 'timestamp']
        df.drop(columns=['timestamp'], inplace=True)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.dropna(inplace=True)

        # Step 3: Pivot matrix
        df_sample = df if len(df) < 10000 else df.sample(n=10000, random_state=42)
        matrix = df_sample.pivot_table(index='userId', columns='productId', values='rating').fillna(0)
        st.write("✅ Matrix shape:", matrix.shape)

        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            st.error("Not enough data to perform clustering.")
            st.stop()

        # Step 4: Scale and PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(matrix)
        n_components = min(30, scaled_data.shape[1])
        if n_components < 2:
            st.error("Not enough product features for PCA.")
            st.stop()

        pca = PCA(n_components=n_components, random_state=42)
        reduced_data = pca.fit_transform(scaled_data)

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

        # Step 5: Clustering
        try:
            kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=512)
            matrix['Cluster_KMeans'] = kmeans.fit_predict(reduced_data)
            kmeans_score = silhouette_score(reduced_data, matrix['Cluster_KMeans'])
        except Exception as e:
            st.warning(f"KMeans failed: {e}")
            kmeans_score = "N/A"

        try:
            hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
            matrix['Cluster_HC'] = hc.fit_predict(reduced_data)
            hc_score = silhouette_score(reduced_data, matrix['Cluster_HC'])
        except Exception as e:
            st.warning(f"Hierarchical clustering failed: {e}")
            hc_score = "N/A"

        # Step 6: Store matrix
        st.session_state['matrix'] = matrix.copy()

        # Step 7: Model selection
        available_models = [col for col in ['Cluster_KMeans', 'Cluster_HC'] if col in matrix.columns]
        if not available_models:
            st.error("No clustering models available.")
            st.stop()

        st.sidebar.header("🔧 Model Selection")
        model_choice = st.sidebar.selectbox("Choose clustering model", available_models)

        # Step 8: Recommendation logic
        def recommend_products(user_id, cluster_label_col, matrix):
            if cluster_label_col not in matrix.columns:
                return f"Model '{cluster_label_col}' not found."
            if user_id not in matrix.index:
                return "User ID not found."

            user_cluster = matrix.loc[user_id, cluster_label_col]
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

        # Step 9: Recommendation section
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

        # Step 10: Model comparison
        model_scores = {
            'Cluster_KMeans': kmeans_score,
            'Cluster_HC': hc_score
        }
        st.subheader("📈 Model Comparison")
        st.dataframe(pd.DataFrame(model_scores.items(), columns=['Model', 'Silhouette Score']))

        st.success("🎉 All steps completed successfully!")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
