""" AD model configuration for algorithm parameters """

import numpy as np

# Configuration below defines AD model search parameter space.
ad_config = {
    # --------------------------------------------------
    # SciKit-Learn Models
    # --------------------------------------------------
    "skl_dbscan": {
        "search_param": "eps",
        "eps": list(np.linspace(0.001, 1.3, 100).round(4)),
        "samples": list(range(7, 10, 1))
    },
    "skl_svm": {
        "search_param": "threshold",
        "threshold": list(np.linspace(0, 1, 100).round(4)),
        "kernel": ["linear"],
        # Optimal kernel is gaussian however due to performance it is changed
        # to linear due to performance
        "C": [0.001],
        "epsilon": [3],
        "degree": [2],
        "gamma": ["auto"],
        "nu": [0.3]
    },
    "skl_sgd": {
        "search_param": "threshold",
        "threshold": list(np.linspace(-1, 10, 51).round(4)) + list(np.linspace(20, 60, 20).round(4)),
        "loss": ["squared_error"], 
        "penalty": ["l1", "l2"],
        "max_iter": [300, 600, 900, 1200],
        "tol": [1e-2, 1e-3, 1e-4, 1e-5],
    },
    "skl_kmeans": {
        "search_param": "threshold",
        "threshold": list(np.linspace(0.01, 1, 200).round(4)),
        "n_clusters": [2],
        "random_state": [[0, 1]],
    },

    "pytorch_autoencoder": {
        "search_param": "threshold",
        "threshold": list(np.linspace(0.01, 5000, 200).round(4)),
        "lr": [0.1],
    },

    # --------------------------------------------------
    # ADEF Custom Models
    # --------------------------------------------------
    # Optimized - AUC: 0.85 - 0.95
    "thb": {
        "search_param": "threshold",
        "threshold": [
            float(f"{x:.4f}")
            for x in list(np.linspace(0, 150, 100))
        ],
    }
}
