import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
data, _ = make_blobs(n_samples=300, centers=3, random_state=42)
k = 3
max_iters = 100
tol = 1e-4
np.random.seed(42)
centroids = data[np.random.choice(data.shape[0], k, replace=False)]
for _ in range(max_iters):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    if np.linalg.norm(new_centroids - centroids) < tol:
        break
    centroids = new_centroids
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
plt.title('K-Means Clustering')
plt.show()
