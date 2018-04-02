import numpy as np
from sklearn.neighbors import NearestNeighbors


def euclidean_distance_matrix(x: np.ndarray, y: np.ndarray):
    dist = np.matmul(x, y.transpose())
    dist *= -2
    x_sqr_sum = (x ** 2).sum(axis=1)
    y_sqr_sum = (y ** 2).sum(axis=1)
    dist += x_sqr_sum[:, np.newaxis]
    dist += y_sqr_sum[np.newaxis, :]
    return np.sqrt(dist)


def cosine_distance_matrix(x: np.ndarray, y: np.ndarray):
    dots = np.matmul(x, y.transpose())
    x_mags = np.sqrt((x ** 2).sum(axis=1))
    y_mags = np.sqrt((y ** 2).sum(axis=1))
    dots /= x_mags[:, np.newaxis]
    dots /= y_mags[np.newaxis, :]
    return 1 - dots


class NearestNeighborDecision:
    _EPSILON = 1e-5

    def __init__(self, y: np.ndarray, neighbours_idx: np.ndarray, neighbour_dist: np.ndarray = None):
        self.idx = neighbours_idx
        self.y = y
        if neighbour_dist is None:
            self.dist = np.ones(self.idx.shape)
        else:
            self.dist = neighbour_dist + NearestNeighborDecision._EPSILON

    def predict(self, neighbours_k: int = None):
        if neighbours_k is None:
            neighbours_k = self.dist.shape[1]
        n_samples = self.idx.shape[0]

        # Make class score indices of class labels
        classes_idx = self.y[self.idx[:, :neighbours_k]]
        first_y = classes_idx.min()
        n_classes = classes_idx.max() - first_y + 1
        classes_idx -= first_y

        # Compute class scores
        scores = np.zeros((n_samples, n_classes))
        np.add.at(scores,
                  [np.arange(n_samples).reshape(-1, 1), classes_idx],
                  1 / self.dist[:, :neighbours_k])

        best_class_idx = scores.argmax(axis=1)
        return best_class_idx + first_y


class MyNearestNeighbors:
    def __init__(self, k: int, metric: str):
        self.k = k
        self.fit_x = None
        self.y = None
        if metric == 'euclidean':
            self.metric = euclidean_distance_matrix
        else:
            self.metric = cosine_distance_matrix

    def fit(self, x: np.ndarray):
        self.fit_x = x

    def kneighbors(self, x: np.ndarray, return_distance: bool):
        dist = self.metric(self.fit_x, x)
        nearest_neighbors_idx = np.argsort(dist, axis=0)[:self.k].copy().transpose()

        if return_distance:
            return dist[nearest_neighbors_idx, np.arange(x.shape[0]).reshape(-1, 1)], nearest_neighbors_idx
        else:
            return nearest_neighbors_idx


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size=None):
        self.block_size = test_block_size
        self.weights = weights
        self.y = None
        self.k = k

        if strategy == 'my_own':
            self.model = MyNearestNeighbors(k, metric)
        else:
            self.model = NearestNeighbors(k, algorithm=strategy, metric=metric)

    def fit(self, X, y=None):
        self.model.fit(X)
        self.y = y
        return self

    def find_kneighbors(self, X: np.ndarray, return_distance):
        if self.block_size is None:
            return self.model.kneighbors(X, return_distance=return_distance)
        else:
            n_samples = X.shape[0]
            n_blocks = n_samples // self.block_size

            if n_samples % self.block_size > 0:
                n_blocks += 1

            if return_distance:
                distances = np.empty((n_samples, self.k), float)
            else:
                distances = None
            indices = np.empty((n_samples, self.k), np.int)

            for i in range(n_blocks):
                block_start = i * self.block_size
                block_end = block_start + self.block_size
                block_end = block_end if block_end <= n_samples else n_samples

                if return_distance:
                    distances[block_start:block_end], indices[block_start:block_end] = \
                        self.model.kneighbors(X[block_start:block_end], return_distance=True)
                else:
                    indices[block_start:block_end] = \
                        self.model.kneighbors(X[block_start:block_end], return_distance=False)

            if return_distance:
                return distances, indices
            return indices

    def predict(self, x: np.ndarray):
        if self.weights:
            dist, no = self.find_kneighbors(x, True)
            predictor = NearestNeighborDecision(self.y, no, dist)
        else:
            idx = self.find_kneighbors(x, False)  # type: np.ndarray
            predictor = NearestNeighborDecision(self.y, idx)
        return predictor.predict()


__all__ = ['KNNClassifier', 'NearestNeighborDecision']
