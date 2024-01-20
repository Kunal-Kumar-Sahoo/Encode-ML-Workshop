import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_digits

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean = None
        self.std_dev = None
        self.components = None

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std_dev = np.std(X, axis=0)
        
        # Handle cases where standard deviation is zero
        self.std_dev[self.std_dev == 0] = 1e-8

        X_normalized = (X - self.mean) / self.std_dev
        
        covariance_matrix = np.cov(X_normalized, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_normalized = (X - self.mean) / self.std_dev
        return np.dot(X_normalized, self.components)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        return np.dot(X_transformed, self.components.T) * self.std_dev + self.mean

# Example usage
digits = load_digits()
X = digits.data
y = digits.target

n_components = 3
pca = PCA(n_components=n_components)
pca.fit(X)
X_transformed = pca.transform(X)

# Create a 3D scatter plot using plotly
fig = go.Figure()

for i in range(10):
    indices = (y == i)
    fig.add_trace(go.Scatter3d(
        x=X_transformed[indices, 0],
        y=X_transformed[indices, 1],
        z=X_transformed[indices, 2],
        mode='markers',
        marker=dict(size=4),
        name=str(i)
    ))

fig.update_layout(
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3'
    ),
    title='PCA of Digits dataset in 3D'
)

fig.show()
