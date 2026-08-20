import os
import time
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utile import load_train_test_split, vectorize_tfidf, save_metrics, save_predictions, SEED, ResourceMonitor

N_STATES = int(os.environ.get("N_STATES", 100))
ALPHA = float(os.environ.get("ALPHA", 0.2))
EPSILON_START = float(os.environ.get("EPSILON_START", 1.0))
EPSILON_MIN = float(os.environ.get("EPSILON_MIN", 0.01))
EPSILON_DECAY = float(os.environ.get("EPSILON_DECAY", 0.9999))
N_EPOCHS = int(os.environ.get("N_EPOCHS", 3))


def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(2)
    return int(np.argmax(Q[state]))


def main():
    start = time.time()
    np.random.seed(SEED)
    monitor = ResourceMonitor()
    monitor.start()

    X_train, X_test, y_train, y_test = load_train_test_split()
    X_train_vec, X_test_vec, _ = vectorize_tfidf(X_train, X_test)

    svd = TruncatedSVD(n_components=50, random_state=SEED)
    X_train_reduced = svd.fit_transform(X_train_vec)
    X_test_reduced = svd.transform(X_test_vec)

    kmeans = KMeans(n_clusters=N_STATES, n_init=10, random_state=SEED)
    train_states = kmeans.fit_predict(X_train_reduced)
    test_states = kmeans.predict(X_test_reduced)

    Q = np.zeros((N_STATES, 2))
    epsilon = EPSILON_START
    y_train_arr = np.array(y_train)
    n_train = len(y_train_arr)
    order_rng = np.random.RandomState(SEED)

    for epoch in range(N_EPOCHS):
        order = order_rng.permutation(n_train)
        for idx in order:
            state = train_states[idx]
            true_label = y_train_arr[idx]
            action = epsilon_greedy(Q, state, epsilon)
            reward = 1.0 if action == true_label else -1.0

            Q[state, action] += ALPHA * (reward - Q[state, action])
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    y_pred = np.argmax(Q[test_states], axis=1)

    q_test = Q[test_states]
    exp_q = np.exp(q_test - q_test.max(axis=1, keepdims=True))
    probs = exp_q / exp_q.sum(axis=1, keepdims=True)
    y_proba = probs[:, 1]

    y_test_arr = np.array(y_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_arr, y_pred, average="binary"
    )

    resource_stats = monitor.stop()
    elapsed = round(time.time() - start, 3)

    metrics = {
        "algorithm": "SARSA-bandit (on news text)",
        "n_states": N_STATES,
        "n_epochs": N_EPOCHS,
        "accuracy": accuracy_score(y_test_arr, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc_score(y_test_arr, y_proba),
        "final_epsilon": round(epsilon, 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_time_seconds": elapsed,
        "throughput_samples_per_second": round((n_train * N_EPOCHS) / elapsed, 2),
        **resource_stats,
    }
    save_metrics("sarsa_bandit", metrics)
    save_predictions("sarsa_bandit", y_test_arr, y_proba)


if __name__ == "__main__":
    main()