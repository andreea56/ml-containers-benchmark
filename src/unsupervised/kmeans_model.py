import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score
)

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utile import load_train_test_split, vectorize_tfidf, save_metrics, SEED, ResourceMonitor


def main():
    start = time.time()
    monitor = ResourceMonitor()
    monitor.start()

    X_train, X_test, y_train, y_test = load_train_test_split()
    X_all = X_train + X_test
    y_all = np.array(y_train + y_test)

    X_vec, _, _ = vectorize_tfidf(X_all, X_all[:1])

    n_components = min(100, X_vec.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    X_reduced = svd.fit_transform(X_vec)

    model = KMeans(n_clusters=2, n_init=10, random_state=SEED)
    cluster_labels = model.fit_predict(X_reduced)

    resource_stats = monitor.stop()
    elapsed = round(time.time() - start, 3)

    metrics = {
        "algorithm": "K-Means",
        "n_clusters": 2,
        "silhouette_score": float(silhouette_score(X_reduced, cluster_labels)),
        "adjusted_rand_index_vs_true_labels": float(adjusted_rand_score(y_all, cluster_labels)),
        "normalized_mutual_info_vs_true_labels": float(normalized_mutual_info_score(y_all, cluster_labels)),
        "n_samples": len(X_all),
        "svd_components": n_components,
        "training_time_seconds": elapsed,
        "throughput_samples_per_second": round(len(X_all) / elapsed, 2),
        **resource_stats,
    }

    save_metrics("kmeans", metrics)


if __name__ == "__main__":
    main()