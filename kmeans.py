import numpy as np


class KMeans:
    def __init__(self, k=3, initializer: str = "_kmeans_pp"):
        self._k = k
        self._centroids = None
        self._x = None
        possible_init_str = ["_kmeans_n", "kmeans_normal", "kmeans", "_kmean_n", "_kmean_normal", "kmean"]
        if initializer in possible_init_str:
            self.initializer = "_kmeans_normal"
        else:
            self.initializer = "_kmeans_pp"
        self._cur_distortion = np.inf

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
    def _kmeans_normal(x: np.ndarray, k: int):
        m = x.shape[0]
        random_idx = np.random.choice(np.arange(m), size=k, replace=False)
        centroids = x[random_idx]
        return centroids

    @staticmethod
    def _kmeans_pp(x: np.ndarray, k: int):
        m = x.shape[0]
        centroids = np.zeros(shape=(k, x.shape[1], 1))
        random_ind = np.random.randint(m)
        centroids[0] = x[random_ind]

        if k > 1:
            dist_tensor = KMeans._get_squared_distance(x, centroids[0])
            centroids[1] = x[np.argmax(dist_tensor)]

        for cur_k in range(2, k):
            temp_centroids = KMeans._reshape_centroids(centroids)
            dist_tensor = KMeans._get_squared_distance(temp_centroids, x)
            min_dist = KMeans._get_min_dist(dist_tensor)
            max_ind = np.argmax(min_dist)
            centroids[cur_k] = x[max_ind]

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
    def _get_min_dist(dist_tensor: np.ndarray):
        min_dist_2d = np.min(dist_tensor, axis=-1)
        min_dist = np.squeeze(min_dist_2d)
        return min_dist

    @staticmethod
    def _get_closest_examples_avg(closest_idx: np.ndarray, cur_k: int, x: np.ndarray):
        cluster_k_idx = np.where(closest_idx == cur_k)[0]
        closest_examples = x[cluster_k_idx]
        if closest_examples.size:
            return np.mean(closest_examples, axis=0, keepdims=True)
        else:
            fake_centroid = np.empty(shape=(1, 1, 1))
            fake_centroid[:] = np.nan
            return fake_centroid

    @staticmethod
    def _get_distortion(x: np.ndarray, centroid: np.ndarray, closest_idx: np.ndarray):
        m = x.shape[0]
        sq_dist = KMeans._get_squared_distance(x, centroid[closest_idx])
        distortion = 1 / m * np.sum(sq_dist)
        return distortion

    def _construct_centroids(self, x: np.ndarray, k: int):
        initializer = eval("self."+self.initializer)
        return initializer(x, k)

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
                self._cur_distortion = least_distortion
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

    def get_distortion(self):
        return self._cur_distortion
