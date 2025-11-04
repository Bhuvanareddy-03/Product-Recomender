import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Clustering Recommender", layout="wide")
st.title("🛍️ Product Recommendation using Clustering")

uploaded_file = st.file_uploader("Upload your ratings CSV file", type=["csv"])
if uploaded_file:
    try:
        # Step 1: Load and clean data
        df = pd.read_csv(uploaded_file)
        if df.shape[1] != 4:
            st.error("Uploaded file must have exactly 4 columns: userId, productId, rating, timestamp.")
            st.stop()

        df.columns = ['userId', 'productId', 'rating', 'timestamp']
        df.drop(columns=['timestamp'], inplace=True)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.dropna(inplace=True)

        st.write("✅ File loaded successfully")
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write(df.head())

        # Step 2: Pivot user-product matrix
        df_sample = df.sample(n=min(10000, len(df)), random_state=42)
        user_product_matrix = df_sample.pivot_table(index='userId',
                                                    columns='productId',
                                                    values='rating').fillna(0)
        st.write("✅ Matrix shape:", user_product_matrix.shape)

        if user_product_matrix.shape[0] < 2 or user_product_matrix.shape[1] < 2:
            st.error("Not enough data to perform clustering.")
            st.stop()

        # Step 3: Scale and PCA
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(user_product_matrix)
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

        # Step 4: KMeans clustering
        try:
            kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=512)
            kmeans_labels = kmeans.fit_predict(reduced_data)
            user_product_matrix['Cluster_KMeans'] = kmeans_labels
            kmeans_score = silhouette_score(reduced_data, kmeans_labels)
        except Exception as e:
            st.warning(f"KMeans clustering failed: {e}")
            kmeans_score = "N/A"

        # Step 5: Hierarchical clustering
        try:
            hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
            hc_labels = hc.fit_predict(reduced_data)
            user_product_matrix['Cluster_HC'] = hc_labels
            hc_score = silhouette_score(reduced_data, hc_labels)
        except Exception as e:
            st.warning(f"Hierarchical clustering failed: {e}")
            hc_score = "N/A"

        # Step 6: Dynamic model selection
        available_models = [col for col in ['Cluster_KMeans', 'Cluster_HC'] if col in user_product_matrix.columns]
        if not available_models:
            st.error("No clustering models available for selection.")
            st.stop()

        st.sidebar.header("🔧 Model Selection")
        model_choice = st.sidebar.selectbox("Choose clustering model", available_models)

        # Step 7: Recommendation logic
        def recommend_products(user_id, cluster_label_col):
            if cluster_label_col not in user_product_matrix.columns:
                return f"Selected model '{cluster_label_col}' did not produce valid clusters."
            if user_id not in user_product_matrix.index:
                return "User ID not found in the data sample."

            user_cluster = user_product_matrix.loc[user_id, cluster_label_col]
            cluster_users = user_product_matrix[user_product_matrix[cluster_label_col] == user_cluster]

            if cluster_users.shape[0] < 2:
                return f"No similar users found in cluster {user_cluster}."

            cluster_users = cluster_users.drop(columns=available_models, errors='ignore')
            cluster_users = cluster_users.apply(pd.to_numeric, errors='coerce')
            cluster_users = cluster_users.dropna(axis=1, how='all')

            if cluster_users.empty:
                return "No product ratings available in this cluster."

            mean_ratings = cluster_users.mean().sort_values(ascending=False)
            if mean_ratings.empty:
                return "No product ratings found after averaging."

            return mean_ratings.head(5)

        # Step 8: Recommendation section
        st.subheader("🎯 Get Recommendations")
        if not user_product_matrix.empty:
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
        else:
            st.warning("User-product matrix is empty. Cannot generate recommendations.")

        # Step 9: Model comparison
        model_scores = {
            'Cluster_KMeans': kmeans_score,
            'Cluster_HC': hc_score
        }
        st.subheader("📈 Model Comparison")
        st.dataframe(pd.DataFrame(model_scores.items(), columns=['Model', 'Silhouette Score']))

        st.success("🎉 All steps completed successfully!")

    except Exception as e:
        st.error(f"❌ App failed at some step: {e}")
