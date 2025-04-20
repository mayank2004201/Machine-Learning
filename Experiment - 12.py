import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
X, _ = make_blobs(n_samples=200, centers=4, cluster_std=1.0, random_state=42)
model = AgglomerativeClustering(n_clusters=4)
labels = model.fit_predict(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set2', s=60)
plt.title("Agglomerative Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
linked = linkage(X, method='ward')
plt.figure(figsize=(12, 5))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=45., leaf_font_size=12.)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Sample Index or Cluster Size")
plt.ylabel("Distance")
plt.grid(True)
plt.show()
