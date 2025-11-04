import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.title("🛍️ Product Recommendation Diagnostic Mode")

uploaded_file = st.file_uploader("Upload your ratings CSV file", type=["csv"])
if uploaded_file:
    try:
        st.write("📥 Step 1: Reading CSV")
        df = pd.read_csv(uploaded_file)
        if df.shape[1] != 4:
            st.error("Uploaded file must have exactly 4 columns: userId, productId, rating, timestamp.")
            st.stop()

        df.columns = ['userId', 'productId', 'rating', 'timestamp']
        df.drop(columns=['timestamp'], inplace=True)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df.dropna(inplace=True)

        st.write("✅ Step 1 complete: File loaded and cleaned")
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write(df.head())

        st.write("📊 Step 2: Creating user-product matrix")
        df_sample = df.sample(n=min(10000, len(df)), random_state=42)
        user_product_matrix = df_sample.pivot_table(index='userId',
                                                    columns='productId',
                                                    values='rating').fillna(0)
        st.write("✅ Step 2 complete: Matrix shape", user_product_matrix.shape)

        if user_product_matrix.shape[0] < 2 or user_product_matrix.shape[1] < 2:
            st.error("Not enough data to perform PCA.")
            st.stop()

        st.write("📈 Step 3: Scaling data")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(user_product_matrix)
        st.write("✅ Step 3 complete: Scaled shape", scaled_data.shape)

        st.write("🧠 Step 4: PCA transformation")
        n_components = min(30, scaled_data.shape[1])
        if n_components < 2:
            st.error("Not enough product features for PCA.")
            st.stop()

        pca = PCA(n_components=n_components, random_state=42)
        reduced_data = pca.fit_transform(scaled_data)
        st.write("✅ Step 4 complete: PCA shape", reduced_data.shape)

        # Explained variance ratio
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

        st.write("🔁 Step 5: KMeans clustering")
        try:
            kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=512)
            kmeans_labels = kmeans.fit_predict(reduced_data)
            user_product_matrix['Cluster_KMeans'] = kmeans_labels
            if len(set(kmeans_labels)) > 1:
                kmeans_score = silhouette_score(reduced_data, kmeans_labels)
            else:
                kmeans_score = "Only one cluster found"
            st.write("✅ KMeans clustering complete")
            st.write("KMeans labels:", np.unique(kmeans_labels))
            st.write("KMeans score:", kmeans_score)
        except Exception as e:
            st.error(f"KMeans failed: {e}")
            st.stop()

        st.write("🔁 Step 6: Hierarchical clustering")
        try:
            hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
            hc_labels = hc.fit_predict(reduced_data)
            user_product_matrix['Cluster_HC'] = hc_labels
            if len(set(hc_labels)) > 1:
                hc_score = silhouette_score(reduced_data, hc_labels)
            else:
                hc_score = "Only one cluster found"
            st.write("✅ Hierarchical clustering complete")
            st.write("HC labels:", np.unique(hc_labels))
            st.write("HC score:", hc_score)
        except Exception as e:
            st.error(f"Hierarchical clustering failed: {e}")
            st.stop()

        st.success("🎉 Clustering steps completed successfully!")

    except Exception as e:
        st.error(f"❌ App failed at some step: {e}")
