import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import seaborn as sns
from numpy.linalg import svd

def get_mysql_data():
    engine = create_engine('mysql+mysqlconnector://book_user:1234@localhost:3306/books_db')
    query = 'SELECT * FROM books_preprocessed'
    df = pd.read_sql(query, engine)
    return df

df = get_mysql_data()
df.fillna(0, inplace=True)

def encode_categorical_features(df, categorical_columns):
    encoded_df = df.copy()
    for column in categorical_columns:
        encoded_df[column] = encoded_df[column].astype('category').cat.codes
    return encoded_df

def categorize_price(price):
    if price <= 850:
        return 0
    elif price > 850 and price <= 1250:
        return 1
    else:
        return 2

df['price_category'] = df['price'].apply(categorize_price)

categorical_columns = ['author', 'publisher', 'genre', 'year', 'binding']
df_encoded = encode_categorical_features(df, categorical_columns)

features_for_clustering = ['year', 'binding', 'pages', 'publisher', 'genre', 'price']
X = df_encoded[features_for_clustering].values

def min_max_normalize(X):
    X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_normalized

X_normalized = min_max_normalize(X)

def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

def compute_distances(X, centroids):
    distances = np.zeros((X.shape[0], len(centroids)))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return distances

def assign_clusters(distances):
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        centroids[i, :] = X[labels == i].mean(axis=0)
    return centroids

def kmeans(X, k, max_iterations=100000):
    centroids = initialize_centroids(X, k)
    
    for _ in range(max_iterations):
        distances = compute_distances(X, centroids)
        
        labels = assign_clusters(distances)
        
        new_centroids = update_centroids(X, labels, k)
        
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

def compute_inertia(X, labels, centroids):
    inertia = 0
    for i, centroid in enumerate(centroids):
        inertia += np.sum((X[labels == i] - centroid) ** 2)
    return inertia

def elbow_method(X, k_range):
    inertias = []
    for k in k_range:
        labels, centroids = kmeans(X, k)
        inertia = compute_inertia(X, labels, centroids)
        inertias.append(inertia)
    return inertias

k_range = range(1, 11) 
inertias = elbow_method(X_normalized, k_range)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.savefig('plots/models_results/elbow_function.png')

optimal_k = 3

labels, centroids = kmeans(X_normalized, optimal_k)

df_encoded['cluster'] = labels

def pca(X, n_components=2):
    X_centered = X - np.mean(X, axis=0)
    
    covariance_matrix = np.cov(X_centered, rowvar=False)
    
    U, S, Vt = svd(covariance_matrix)
    
    components = Vt[:n_components]
    
    X_reduced = np.dot(X_centered, components.T)
    
    return X_reduced

X_reduced = pca(X_normalized, n_components=2)

df_encoded['PCA1'] = X_reduced[:, 0]
df_encoded['PCA2'] = X_reduced[:, 1]

centroids_pca = pca(centroids, n_components=2)

plt.figure(figsize=(12, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=df_encoded, palette='viridis', s=100)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title(f'K-means Clustering with k={optimal_k} and PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.savefig('plots/models_results/k_means.png')