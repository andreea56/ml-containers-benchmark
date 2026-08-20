import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utile import load_train_test_split, vectorize_tfidf, save_metrics, save_predictions, SEED, ResourceMonitor


def main():
    start = time.time()
    monitor = ResourceMonitor()
    monitor.start()

    X_train, X_test, y_train, y_test = load_train_test_split()
    X_train_vec, X_test_vec, _ = vectorize_tfidf(X_train, X_test)

    model = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, random_state=SEED, solver="lbfgs")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )

    resource_stats = monitor.stop()
    elapsed = round(time.time() - start, 3)

    metrics = {
        "algorithm": "Logistic Regression",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_time_seconds": elapsed,
        "throughput_samples_per_second": round(len(X_train) / elapsed, 2),
        **resource_stats,
    }

    save_metrics("logistic_regression", metrics)
    save_predictions("logistic_regression", y_test, y_proba)


if __name__ == "__main__":
    main()