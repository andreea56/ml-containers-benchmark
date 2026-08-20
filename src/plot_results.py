import os
import json
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
SUPERVISED_ALGOS = ["svm", "random_forest", "logistic_regression", "deeptext",
                    "sarsa_bandit", "dqn_bandit"]
SUPERVISED_LABELS = {
    "svm": "SVM",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "deeptext": "DeepText (CNN)",
    "sarsa_bandit": "SARSA (bandit)",
    "dqn_bandit": "DQN (bandit)",
}
UNSUPERVISED_ALGOS = ["kmeans", "dbscan"]
UNSUPERVISED_LABELS = {"kmeans": "K-Means", "dbscan": "DBSCAN"}
REINFORCEMENT_ALGOS = ["sarsa", "dqn"]
REINFORCEMENT_LABELS = {"sarsa": "SARSA (FrozenLake)", "dqn": "DQN (CartPole)"}
ALL_ALGO_LABELS = {**SUPERVISED_LABELS, **UNSUPERVISED_LABELS, **REINFORCEMENT_LABELS}
ALL_ALGOS = SUPERVISED_ALGOS + UNSUPERVISED_ALGOS + REINFORCEMENT_ALGOS

PARADIGM_SUPERVISED = ["svm", "random_forest", "logistic_regression", "deeptext"]
PARADIGM_UNSUPERVISED = ["kmeans", "dbscan"]
PARADIGM_REINFORCEMENT = ["sarsa", "dqn", "sarsa_bandit", "dqn_bandit"]
PARADIGM_COLOR_MAP = {
    **{a: "#4c72b0" for a in PARADIGM_SUPERVISED},
    **{a: "#dd8452" for a in PARADIGM_UNSUPERVISED},
    **{a: "#55a868" for a in PARADIGM_REINFORCEMENT},
}


def load_aggregated():
    path = os.path.join(OUTPUT_DIR, "aggregated_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run aggregate_results.py first.")
    with open(path) as f:
        return json.load(f)


def find_representative_predictions(algo):
    pattern = os.path.join(OUTPUT_DIR, f"{algo}_seed*_predictions.npz")
    matches = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"_seed(\d+)_", p).group(1))
    )
    if not matches:
        return None
    data = np.load(matches[0])
    seed = int(re.search(r"_seed(\d+)_", matches[0]).group(1))
    return data["y_test"], data["y_proba"], seed


def plot_roc_curves(aggregated):
    available = [a for a in SUPERVISED_ALGOS if find_representative_predictions(a) is not None]
    if not available:
        print("No prediction files found - skipping ROC curves.")
        return
    n = len(available)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax, algo in zip(axes_flat, available):
        y_test, y_proba, seed = find_representative_predictions(algo)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_summary = aggregated.get(algo, {}).get("roc_auc", {})
        ax.plot(fpr, tpr, color="black", linewidth=1.3)
        ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1)
        ax.set_title(SUPERVISED_LABELS[algo], fontsize=11, color="#b5651d")
        ax.set_xlabel("1 - specificity", color="#1f4e8c")
        ax.set_ylabel("sensitivity", color="#1f4e8c")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        if auc_summary:
            ax.text(0.5, 0.08, f"AUC = {auc_summary['mean']:.3f}", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
    for ax in axes_flat[len(available):]:
        ax.axis("off")
    fig.suptitle("ROC curves - all classification-style algorithms", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "roc_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_supervised_bar(aggregated):
    labels = [a for a in SUPERVISED_ALGOS if a in aggregated]
    if not labels:
        print("No supervised/bandit aggregated results found - skipping bar chart.")
        return
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_display = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (key, display) in enumerate(zip(metric_keys, metric_display)):
        means = [aggregated[a].get(key, {}).get("mean", 0) for a in labels]
        stds = [aggregated[a].get(key, {}).get("std", 0) for a in labels]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=display)
    ax.set_xticks(x + width * (len(metric_keys) - 1) / 2)
    ax.set_xticklabels([SUPERVISED_LABELS[a] for a in labels], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score (mean +/- std across seeds)")
    ax.set_title("Classification-style algorithms - performance comparison")
    ax.legend(loc="lower right", fontsize=8, ncol=5)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "supervised_metrics_bar.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_unsupervised_bar(aggregated):
    labels = [a for a in UNSUPERVISED_ALGOS if a in aggregated]
    if not labels:
        print("No unsupervised aggregated results found - skipping bar chart.")
        return
    metric_keys = ["adjusted_rand_index_vs_true_labels", "normalized_mutual_info_vs_true_labels", "silhouette_score"]
    metric_display = ["Adjusted Rand Index", "Normalized Mutual Info", "Silhouette score"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (key, display) in enumerate(zip(metric_keys, metric_display)):
        means = [aggregated[a].get(key, {}).get("mean", 0) for a in labels]
        stds = [aggregated[a].get(key, {}).get("std", 0) for a in labels]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=display)
    ax.set_xticks(x)
    ax.set_xticklabels([UNSUPERVISED_LABELS[a] for a in labels])
    ax.set_ylabel("score (mean +/- std across seeds)")
    ax.set_title("Unsupervised learning - clustering quality")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "unsupervised_metrics_bar.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_reinforcement_bar(aggregated):
    labels = [a for a in REINFORCEMENT_ALGOS if a in aggregated]
    if not labels:
        print("No reinforcement aggregated results found - skipping bar chart.")
        return
    means, stds = [], []
    for algo in labels:
        key = "success_rate_last_500_episodes" if algo == "sarsa" else "avg_reward_last_50_episodes"
        summary = aggregated[algo].get(key, {})
        means.append(summary.get("mean", 0))
        stds.append(summary.get("std", 0))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar([REINFORCEMENT_LABELS[a] for a in labels], means, yerr=stds, capsize=4,
           color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("average reward / success rate (mean +/- std)")
    ax.set_title("Reinforcement learning - performance (native environments)")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "reinforcement_reward_bar.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_training_time(aggregated):
    labels, means, stds, colors = [], [], [], []
    for algo in ALL_ALGOS:
        if algo not in aggregated:
            continue
        summary = aggregated[algo].get("training_time_seconds", {})
        labels.append(ALL_ALGO_LABELS[algo])
        means.append(summary.get("mean", 0))
        stds.append(summary.get("std", 0))
        colors.append(PARADIGM_COLOR_MAP[algo])
    if not labels:
        print("No aggregated results found - skipping training time chart.")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, means, yerr=stds, capsize=4, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("training time in seconds (mean +/- std, log scale)")
    ax.set_title("Training time across all algorithms")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    legend_handles = [
        mpatches.Patch(color="#4c72b0", label="Supervised learning"),
        mpatches.Patch(color="#dd8452", label="Unsupervised learning"),
        mpatches.Patch(color="#55a868", label="Reinforcement learning (native + bandit)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "training_time_bar.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_resource_usage(aggregated):
    labels, cpu_means, mem_means, colors = [], [], [], []
    for algo in ALL_ALGOS:
        if algo not in aggregated:
            continue
        cpu = aggregated[algo].get("cpu_percent_mean", {}).get("mean")
        mem = aggregated[algo].get("memory_mb_max", {}).get("mean")
        if cpu is None or mem is None:
            continue
        labels.append(ALL_ALGO_LABELS[algo])
        cpu_means.append(cpu)
        mem_means.append(mem)
        colors.append(PARADIGM_COLOR_MAP[algo])
    if not labels:
        print("No resource usage data found - skipping resource chart.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(labels, cpu_means, color=colors)
    ax1.set_ylabel("mean CPU usage (%)")
    ax1.set_title("CPU utilization")
    ax1.tick_params(axis="x", rotation=40)
    for tick in ax1.get_xticklabels():
        tick.set_ha("right")
    ax1.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax2.bar(labels, mem_means, color=colors)
    ax2.set_ylabel("peak memory (MB)")
    ax2.set_title("Memory consumption")
    ax2.tick_params(axis="x", rotation=40)
    for tick in ax2.get_xticklabels():
        tick.set_ha("right")
    ax2.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    legend_handles = [
        mpatches.Patch(color="#4c72b0", label="Supervised learning"),
        mpatches.Patch(color="#dd8452", label="Unsupervised learning"),
        mpatches.Patch(color="#55a868", label="Reinforcement learning (native + bandit)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))

    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "resource_usage_bar.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    aggregated = load_aggregated()
    plot_roc_curves(aggregated)
    plot_supervised_bar(aggregated)
    plot_unsupervised_bar(aggregated)
    plot_reinforcement_bar(aggregated)
    plot_training_time(aggregated)
    plot_resource_usage(aggregated)
    print(f"\nAll available plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()