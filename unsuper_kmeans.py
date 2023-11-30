import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
#!pip install --upgrade kneed
from kneed import KneeLocator

features, true_labels = make_blobs(
    n_samples=200, centers=3, cluster_std=2.75, random_state=42)
features[:5]
true_labels[:5]
features.shape
plt.scatter(features[:,0],features[:,1], )
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]
plt.scatter(scaled_features[:,0],scaled_features[:,1], )
kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)
kmeans.inertia_
kmeans.cluster_centers_
kmeans.labels_[:5]
kwargs = {"init": "random","n_init": 100,"max_iter": 300,"random_state": 42}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
print(kl.elbow)

# A list holds the silhouette coefficients for each k
silhouette_coefficients = []

# Notice we start at 2 clusters for silhouette coefficient
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)
print(silhouette_coefficients)
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()
