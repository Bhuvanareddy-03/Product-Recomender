This Streamlit application provides a scalable solution for product recommendation using hierarchical clustering. 
Designed for data scientists and developers working with user-product rating data, the app allows users to upload a CSV file containing columns such as `userid`, `productid`, `rating`, and optionally `timestamp`. Once uploaded, the app performs preprocessing, dimensionality reduction using PCA, and clustering via Agglomerative Clustering.
It then calculates the silhouette score to evaluate the quality of clustering, helping users understand how well-separated the clusters are. 
The interface includes a data overview section that displays row and column counts, missing values, and a sample of the dataset for quick inspection. 
Users can select a specific user ID from a dropdown menu to receive personalized product recommendations based on the average ratings of other users within the same cluster. 
This approach ensures that recommendations are grounded in collaborative behavior patterns. 
The app is built with Python, leveraging libraries such as pandas, NumPy, scikit-learn, and Streamlit, and is ideal for prototyping, internal demos, or deploying lightweight recommender systems.

link:https://appuct-recomender-olpqben5v4irrgrwi9laxu.streamlit.app/
