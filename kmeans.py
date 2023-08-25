import numpy as np


class KMeans:
    def __init__(self, k=3):
        self._k = k
        self._centroids = None
        self._x = None

    @staticmethod
    def _create_input(x: any):
        new_x = np.array(x, dtype=np.float32)
        new_x = np.expand_dims(new_x, axis=-1)
        return new_x

    @staticmethod
    def _check_dim(x: np.ndarray):
        assert len(x.shape) == 3

    @staticmethod
    def _check_m(x: np.ndarray, k: int):
        assert x.shape[0] >= k

    @staticmethod
    def _construct_centroids(x: np.ndarray, k: int):
        m = x.shape[0]
        random_idx = np.random.choice(np.arange(m), size=k, replace=False)
        centroids = x[random_idx]
        return centroids

    @staticmethod
    def _get_squared_distance(a: np.ndarray, b: np.ndarray):
        return np.sum((a - b) ** 2, axis=1, keepdims=True)

    @staticmethod
    def _reshape_centroids(centroids: np.ndarray):
        new_centroids = centroids.T
        return new_centroids

    @staticmethod
    def _get_closest_idx(dist_tensor: np.ndarray):
        closest_idx_2d = np.argmin(dist_tensor, axis=-1)
        closest_idx = np.squeeze(closest_idx_2d)
        return closest_idx

    @staticmethod
    def _get_closest_examples_avg(closest_idx: np.ndarray, cur_k: int, x: np.ndarray):
        cluster_k_idx = np.where(closest_idx == cur_k)[0]
        closest_examples = x[cluster_k_idx]
        return np.mean(closest_examples, axis=0, keepdims=True)

    @staticmethod
    def _get_distortion(x: np.ndarray, centroid: np.ndarray, closest_idx: np.ndarray):
        m = x.shape[0]
        sq_dist = KMeans._get_squared_distance(x, centroid[closest_idx])
        distortion = 1 / m * np.sum(sq_dist)
        return distortion

    def fit(self, x, steps=1, verbose=True):
        self._x = self._create_input(x)

        self._check_dim(self._x)
        self._check_m(self._x, self._k)

        least_distortion = np.inf
        target_best_idx = None
        best_idx = None

        for step in range(steps):

            if verbose:
                print("Step:", step+1, end="\n\t", flush=True)

            prev_distortion = np.inf
            cur_distortion = 0

            cluster_steps = 0

            _centroids = self._construct_centroids(self._x, self._k)

            while prev_distortion > cur_distortion:

                reshaped_centroids = self._reshape_centroids(_centroids)

                sq_dist = self._get_squared_distance(self._x, reshaped_centroids)

                best_idx = self._get_closest_idx(sq_dist)

                prev_distortion = self._get_distortion(self._x, _centroids, best_idx)
                for k in range(self._k):
                    mean = self._get_closest_examples_avg(best_idx, k, self._x)
                    _centroids[k] = mean

                cur_distortion = self._get_distortion(self._x, _centroids, best_idx)
                cluster_steps += 1

            _centroids_updated = False
            if cur_distortion < least_distortion:
                self._centroids = _centroids
                least_distortion = cur_distortion
                target_best_idx = best_idx
                _centroids_updated = True

            if verbose:
                print("Distortion (Cost):", cur_distortion,
                      "\n\tstep(s): ", cluster_steps,
                      "\n\tcentroids_updated: ", _centroids_updated,
                      flush=True)

        return target_best_idx

    def get_centroids(self):
        if self._centroids is None:
            raise Exception("The KMean must be trained first before getting the centroids")
        simple_centroids = np.squeeze(self._centroids, axis=-1)
        return simple_centroids

    def predict(self, x):
        _x = self._create_input(x)

        self._check_dim(self._x)

        reshaped_centroids = self._reshape_centroids(self._centroids)

        sq_dist = self._get_squared_distance(_x, reshaped_centroids)

        best_idx = self._get_closest_idx(sq_dist)

        return best_idx
