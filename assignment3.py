from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasetG
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# Stratified split: Train, Validation, Test (60%, 20%, 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# k-NN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

# k-fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn, X_train, y_train, cv=skf)

print("Cross Validation Scores: ", cv_scores)
print("Average CV Score: ", np.mean(cv_scores))

# Agglomerative Clustering with Euclidean distance
agglo_euclidean = AgglomerativeClustering(n_clusters=40, metric='euclidean', linkage='ward')
clusters_euclidean = agglo_euclidean.fit_predict(X_train)

# Calculate Silhouette Score for Euclidean Distance
silhouette_euclidean = silhouette_score(X_train, clusters_euclidean)
print(f'Silhouette Score (Euclidean Distance): {silhouette_euclidean}')

# Agglomerative Clustering with Minkowski distance (using average linkage)
agglo_minkowski = AgglomerativeClustering(n_clusters=40, metric='minkowski', linkage='average')
clusters_minkowski = agglo_minkowski.fit_predict(X_train)

# Calculate Silhouette Score for Minkowski Distance
silhouette_minkowski = silhouette_score(X_train, clusters_minkowski)
print(f'Silhouette Score (Minkowski Distance): {silhouette_minkowski}')

# Cosine similarity computation
cosine_sim = cosine_similarity(X_train)

# Agglomerative Clustering with Cosine similarity (using precomputed distances)
agglo_cosine = AgglomerativeClustering(n_clusters=40, metric='precomputed', linkage='average')
clusters_cosine = agglo_cosine.fit_predict(1 - cosine_sim)  # Convert similarity to distance (1 - similarity)

# Calculate Silhouette Score for Cosine Similarity
silhouette_cosine = silhouette_score(X_train, clusters_cosine)
print(f'Silhouette Score (Cosine Similarity): {silhouette_cosine}')

# Discussion of discrepancies between the results
if silhouette_euclidean != silhouette_minkowski or silhouette_euclidean != silhouette_cosine:
    print("Discrepancies observed in the clustering results with different distance metrics.")
else:
    print("No significant discrepancies observed between clustering results.")

# ----------------------------
# Training k-NN Classifier on Euclidean clusters
# ----------------------------

# Using the clusters from Euclidean distance as labels
X_clustered = clusters_euclidean.reshape(-1, 1)  # Reshaping to fit the classifier

# Train k-NN classifier on clustered data with k-fold cross-validation
cv_scores_clustered = cross_val_score(knn, X_clustered, y_train, cv=skf)

print("Cross Validation Scores on Clustered Data: ", cv_scores_clustered)
print("Average CV Score on Clustered Data: ", np.mean(cv_scores_clustered))
