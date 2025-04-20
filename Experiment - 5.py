import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Step 2: Standardization
# Centering the data (subtracting the mean)
X_meaned = X - np.mean(X, axis=0)

# Scaling to unit variance
X_std = X_meaned / np.std(X_meaned, axis=0)

# Step 3: Covariance Matrix Computation
covariance_matrix = np.cov(X_std, rowvar=False)

# Step 4: Eigenvalue and Eigenvector Computation
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Step 5: Selecting Principal Components
# Sort the eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Select the top 2 eigenvectors (principal components)
n_components = 2
eigenvector_subset = sorted_eigenvectors[:, :n_components]

# Step 6: Transforming the Data
X_pca = X_std.dot(eigenvector_subset)

# Create a DataFrame for easier plotting
df_pca = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
df_pca['Target'] = y

# Step 7: Visualize the PCA results
plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(df_pca.loc[df_pca['Target'] == i, 'Principal Component 1'],
                df_pca.loc[df_pca['Target'] == i, 'Principal Component 2'],
                color=color, alpha=.8, lw=lw, label=target_name)

plt.title('PCA of IRIS Dataset (Manual Implementation)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels
target_names = iris.target_names

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Perform PCA
pca = PCA(n_components=2)  # We want to reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

# Step 4: Create a DataFrame for easier plotting
df_pca = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
df_pca['Target'] = y

# Step 5: Visualize the PCA results
plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(df_pca.loc[df_pca['Target'] == i, 'Principal Component 1'],
                df_pca.loc[df_pca['Target'] == i, 'Principal Component 2'],
                color=color, alpha=.8, lw=lw, label=target_name)

plt.title('PCA of IRIS Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()