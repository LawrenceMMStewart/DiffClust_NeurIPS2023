import time
import warnings

import numpy as np
import matplotlib.pyplot as plt


from jax.nn import one_hot
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

import numpy as np
import jax
import matplotlib.pyplot as plt
from forests import pairwise_square_distance, kruskals


'''
Evaluates different clustering methods with a variety
of metrics, on toy data sets.
'''

def extract_labels(Mem):
    """
    extract the labels from a membership matrix
    """
    N = len(Mem)
    Y = np.ones(N) * -1

    # list of points that have not yet been processed
    processed = []
    to_proc = [i for i in range(N) if i not in processed]
    cur_label = 0
    while len(to_proc) > 0:
        # consider an unprocessed point and find all points in the same cluster
        cur_point = to_proc[0]
        ids = np.where(Mem[cur_point] == 1)[0].tolist()
        # set all points in cluster to have a label, and increase the label value for next cluster
        Y[ids] = cur_label
        cur_label += 1
        # add cluster to list of points that have been processed
        processed += ids
        # regenerate list of points that need to be considered
        to_proc = [i for i in range(N) if i not in processed]
    return Y


class Krusk():
    def __init__(self, ncc=2):
        self.ncc=ncc
    def fit(self, X):
        D = pairwise_square_distance(X)
        A, M = kruskals(D, ncc=self.ncc)
        labels = extract_labels(M)
        self.labels_ = labels
        return labels


np.random.seed(0)



# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 250
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
# noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=50, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# ============
# Set up cluster parameters
# ============
# plt.figure(figsize=(9 * 2 + 3, 13))
# plt.subplots_adjust(
#     left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
# )
fig, ax = plt.subplots()

plot_num = 1

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

datasets = [
    (
        noisy_circles,
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    # (
        # varied,
        # {
        #     "eps": 0.18,
        #     "n_neighbors": 2,
        #     "min_samples": 7,
        #     'n_clusters': 3,
        #     "xi": 0.01,
        #     "min_cluster_size": 0.2,
        # },
    # ),

    (blobs, {"n_clusters" : 3, "min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    (
        aniso,
        {
            "n_clusters" : 3,
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    # (no_structure, {}),
]

fig, axes = plt.subplots(len(datasets), 4, figsize=(14, 9))
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params["n_neighbors"], include_self=False
    )
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    kms = cluster.KMeans(n_clusters=params['n_clusters'])
    two_means = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"], n_init="auto")
    ward = cluster.AgglomerativeClustering(
        n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    )
    spectral = cluster.SpectralClustering(
        n_clusters=params["n_clusters"],
        eigen_solver="arpack",
        affinity="nearest_neighbors",
    )
    dbscan = cluster.DBSCAN(eps=params["eps"])
    optics = cluster.OPTICS(
        min_samples=params["min_samples"],
        xi=params["xi"],
        min_cluster_size=params["min_cluster_size"],
    )
    affinity_propagation = cluster.AffinityPropagation(
        damping=params["damping"], preference=params["preference"], random_state=0
    )
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        n_clusters=params["n_clusters"],
        connectivity=connectivity,
    )
    birch = cluster.Birch(n_clusters=params["n_clusters"])
    gmm = mixture.GaussianMixture(
        n_components=params["n_clusters"], covariance_type="full"
    )
    krusk = Krusk(ncc=params['n_clusters'])

    clustering_algorithms = (
        # ("MiniBatch\nKMeans", two_means),
        # ("Affinity\nPropagation", affinity_propagation),
        # ("Spectral\nClustering", spectral),
        # ("Ward", ward),
        # ("Agglomerative\nClustering", average_linkage),
        # ("DBSCAN", dbscan),
        # ("BIRCH", birch),
        # ("OPTICS", optics),
        ("Kruskal's", krusk),
        ("Kmeans", kms),
        ("MeanShift", ms),
        ("GM", gmm),
    )

    for j_algorithm, (name, algorithm) in enumerate(clustering_algorithms):
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category=UserWarning,
            )
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        ax = axes[i_dataset, j_algorithm]
        # ax.plot(len(datasets), len(clustering_algorithms), plot_num)

        yhot_pred = one_hot(y_pred, params['n_clusters'])
        yhot_true = one_hot(y, params['n_clusters'])

        M_pred = yhot_pred @ yhot_pred.T
        M_true = yhot_true @ yhot_true.T
        l2 = np.mean(np.sqrt((M_pred - M_true) ** 2))


        nmi = normalized_mutual_info_score(y_pred, y)
        sil= silhouette_score(X, y_pred)

        box_props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9)
        text = f'NMI: {nmi:.2f}, SC: {sil:.2f}, CoL2: {l2:.2f}'  # Change this to the desired text or value
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11, verticalalignment="top", bbox=box_props)


        if i_dataset == 0:
            ax.set_title(name, size=18)

        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        ax.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        # plt.xlim(-2.5, 2.5)
        # plt.ylim(-2.5, 2.5)
        ax.set_xticks(())
        ax.set_yticks(())
        # plt.text(
        #     0.99,
        #     0.01,
        #     ("%.2fs" % (t1 - t0)).lstrip("0"),
        #     transform=plt.gca().transAxes,
        #     size=15,
        #     horizontalalignment="right",
        # )
        plot_num += 1

plt.show()

