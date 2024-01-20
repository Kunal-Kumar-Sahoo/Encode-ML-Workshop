import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, k:int, max_iterations:int=100) -> None:
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, data:pd.DataFrame) -> None:
        indices = np.random.choice(len(data), self.k, replace=False)
        self.centroids = data.iloc[indices].to_numpy()
    
    def assign_labels(self, data:pd.DataFrame) -> None:
        distances = np.linalg.norm(data.values[:, np.newaxis] - self.centroids, axis=2)
        self.labels = np.argmin(distances, axis=1)

    def update_centroids(self, data: pd.DataFrame) -> None:
        for i in range(self.k):
            self.centroids[i] = np.mean(data[self.labels == i].values, axis=0)
    
    def fit(self, data:pd.DataFrame) -> None:
        self.initialize_centroids(data)

        for _ in range(self.max_iterations):
            old_centroids = np.copy(self.centroids)
            self.assign_labels(data)
            self.update_centroids(data)

            if np.array_equal(old_centroids, self.centroids):
                break

    def predict(self, data):
        distances = np.linalg.norm(data.values[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
if __name__ == '__main__':
    df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/datasets/master/Mall_Customers.csv')
    df = df.iloc[:, [3, 4]]
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.show()

    kmeans = KMeans(5)
    kmeans.fit(df)
    preds = kmeans.predict(df)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=preds)
    plt.show()

