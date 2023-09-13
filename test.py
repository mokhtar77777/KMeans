from kmeans import KMeans
import numpy as np
import matplotlib.pyplot as plt

x = np.array([
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [100, 100],
    [2, 2]
])

kmeans = KMeans(k=4)
kmeans.fit(x)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

print(kmeans.get_centroids())
