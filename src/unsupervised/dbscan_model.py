import time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
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
    n_components = min(50, X_vec.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    X_reduced = svd.fit_transform(X_vec)
    eps = float(os.environ.get("DBSCAN_EPS", 1.5))
    min_samples = int(os.environ.get("DBSCAN_MIN_SAMPLES", 10))
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
    cluster_labels = model.fit_predict(X_reduced)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int(np.sum(cluster_labels == -1))

    # Silhouette score—can only be calculated if there are at least 2 true clusters
    # (we exclude noise points, labeled -1, from the calculation)
    non_noise_mask = cluster_labels != -1
    unique_clusters = set(cluster_labels[non_noise_mask])
    if len(unique_clusters) >= 2:
        silhouette = float(silhouette_score(X_reduced[non_noise_mask], cluster_labels[non_noise_mask]))
    else:
        silhouette = None

    resource_stats = monitor.stop()
    elapsed = round(time.time() - start, 3)
    metrics = {
        "algorithm": "DBSCAN",
        "eps": eps,
        "min_samples": min_samples,
        "clusters_found": n_clusters,
        "noise_points": n_noise,
        "noise_ratio": round(n_noise / len(X_all), 4),
        "silhouette_score": silhouette,
        "adjusted_rand_index_vs_true_labels": float(adjusted_rand_score(y_all, cluster_labels)),
        "normalized_mutual_info_vs_true_labels": float(normalized_mutual_info_score(y_all, cluster_labels)),
        "n_samples": len(X_all),
        "svd_components": n_components,
        "training_time_seconds": elapsed,
        "throughput_samples_per_second": round(len(X_all) / elapsed, 2),
        **resource_stats,
    }
    save_metrics("dbscan", metrics)

if __name__ == "__main__":
    main()