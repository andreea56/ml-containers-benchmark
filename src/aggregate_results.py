import os
import re
import json
import glob
import csv
from collections import defaultdict

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")

SKIP_KEYS = {"algorithm", "seed", "environment"}


def mean(values):
    return sum(values) / len(values) if values else None


def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def collect_runs():
    runs_by_algo = defaultdict(list)
    pattern = re.compile(r"^(.*)_seed(\d+)_metrics\.json$")

    for path in glob.glob(os.path.join(OUTPUT_DIR, "*_seed*_metrics.json")):
        fname = os.path.basename(path)
        match = pattern.match(fname)
        if not match:
            continue
        algo = match.group(1)
        with open(path) as f:
            runs_by_algo[algo].append(json.load(f))

    return runs_by_algo


def aggregate(runs_by_algo):
    aggregated = {}
    for algo, runs in runs_by_algo.items():
        if not runs:
            continue
        numeric_keys = [
            k for k, v in runs[0].items()
            if k not in SKIP_KEYS and isinstance(v, (int, float))
        ]
        algo_summary = {
            "n_runs": len(runs),
            "seeds": sorted(r.get("seed") for r in runs),
        }
        for key in numeric_keys:
            values = [r[key] for r in runs if key in r]
            algo_summary[key] = {
                "mean": round(mean(values), 4),
                "std": round(std(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }
        aggregated[algo] = algo_summary
    return aggregated


def write_json(aggregated):
    out_path = os.path.join(OUTPUT_DIR, "aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved {out_path}")


def write_csv(aggregated):
    out_path = os.path.join(OUTPUT_DIR, "aggregated_results.csv")
    all_metrics = sorted({
        metric for summary in aggregated.values()
        for metric in summary if metric not in ("n_runs", "seeds")
    })

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["algorithm", "n_runs"] + [f"{m}_mean" for m in all_metrics] + [f"{m}_std" for m in all_metrics]
        writer.writerow(header)
        for algo, summary in sorted(aggregated.items()):
            row = [algo, summary["n_runs"]]
            row += [summary.get(m, {}).get("mean", "") for m in all_metrics]
            row += [summary.get(m, {}).get("std", "") for m in all_metrics]
            writer.writerow(row)
    print(f"Saved {out_path}")


def main():
    runs_by_algo = collect_runs()
    if not runs_by_algo:
        print(f"No *_seed*_metrics.json files found in {OUTPUT_DIR}.")
        return

    for algo, runs in runs_by_algo.items():
        print(f"{algo}: {len(runs)} run(s), seeds={sorted(r.get('seed') for r in runs)}")

    aggregated = aggregate(runs_by_algo)
    write_json(aggregated)
    write_csv(aggregated)


if __name__ == "__main__":
    main()