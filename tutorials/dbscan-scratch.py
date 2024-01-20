import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from typing import List

class DBSCAN:
    def __init__(self, eps: float, min_samples: int):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.linalg.norm(point1 - point2)

    def _get_neighbors(self, data: np.ndarray, index: int) -> List[int]:
        distances = [self._euclidean_distance(data[index], data[i]) for i in range(len(data))]
        return [i for i, distance in enumerate(distances) if distance <= self.eps and i != index]

    def _expand_cluster(self, data: np.ndarray, index: int, neighbors: List[int], cluster_id: int):
        self.labels[index] = cluster_id

        for neighbor in neighbors:
            if self.labels[neighbor] == -1:
                self.labels[neighbor] = cluster_id
            elif self.labels[neighbor] == 0:
                self.labels[neighbor] = cluster_id
                new_neighbors = self._get_neighbors(data, neighbor)

                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)

    def fit(self, data: np.ndarray):
        self.labels = np.zeros(len(data), dtype=int)
        cluster_id = 0

        for i in range(len(data)):
            if self.labels[i] != 0:
                continue

            neighbors = self._get_neighbors(data, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = -1
            else:
                cluster_id += 1
                self._expand_cluster(data, i, neighbors, cluster_id)

def plot_clusters(data: np.ndarray, labels: List[int], title: str):
    unique_labels = np.unique(labels)

    plt.figure(figsize=(8, 6))

    for label in unique_labels:
        if label == -1:
            color = 'black'  # Noise points
        else:
            color = plt.cm.jet(label / len(unique_labels))  # Assign a color based on the cluster ID

        cluster_points = data[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {label}')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Example usage
# Generate a synthetic 2D dataset for visualization
np.random.seed(42)

X, _ = make_moons(n_samples=500, noise=0.1, random_state=42)
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.show()

eps = 0.2
min_samples = 5

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X)

plot_clusters(X, dbscan.labels, title='DBSCAN Clustering')