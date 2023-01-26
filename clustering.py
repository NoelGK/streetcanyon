import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
plt.style.use("seaborn")
seed = 786454875
np.random.seed(seed)


# # # Study area
Npoints = 100
xmin, xmax = 0.0, 10.0
area = np.concatenate([np.random.uniform(xmin, xmax, (Npoints, 1)), np.random.uniform(xmin, xmax, (Npoints, 1))], axis=1)

# # # Create synthetic data
N_devs = 25
idx = np.random.randint(0, Npoints, N_devs)
dev_coords = area[idx, :]
LAday = np.random.randint(0, 80, (N_devs, 1))

# plt.scatter(dev_coords[:, 0], dev_coords[:, 1], c=LAday)
# plt.xlim(0, 10)
# plt.ylim(0, 10)
# plt.colorbar()
# plt.title('Location of devices')
# plt.show()

# # # KMeans
K = 4
kmeans = KMeans(n_clusters=K, max_iter=400, random_state=seed).fit(LAday)
labels, centorids = kmeans.labels_, kmeans.cluster_centers_


# # # Scatter clustered points
LA_clusters = [LAday[labels==lab] for lab in np.unique(labels)]
Coord_clusters = [dev_coords[labels==lab, :] for lab in np.unique(labels)]
markers = [".", "^", "+", "d"]
plot_labels = {i: mark for i, mark in enumerate(markers)}

# for coord, la, mark in zip(Coord_clusters, LA_clusters, plot_labels):
#     plt.scatter(coord[:, 0], coord[:, 1], marker=markers[mark], c=la, label=mark)

# plt.legend()
# plt.show()

for cluster in LA_clusters:
    print(cluster)
    print(np.mean(cluster))
