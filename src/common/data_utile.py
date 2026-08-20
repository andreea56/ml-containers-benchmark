import os
import json
import re
import time
import threading
import psutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.environ.get("DATA_DIR", "/data")
TRUE_CSV = os.environ.get("TRUE_CSV", "true.csv")
FAKE_CSV = os.environ.get("FAKE_CSV", "fake.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
SEED = int(os.environ.get("SEED", 42))


class ResourceMonitor:
    """Samples CPU% and RAM (MB) of the current process every `interval`
    seconds, on a background thread, while the algorithm runs. Use as:
        monitor = ResourceMonitor()
        monitor.start()
        ... training code ...
        stats = monitor.stop()
    """
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._process = psutil.Process(os.getpid())
        self._samples_cpu = []
        self._samples_mem = []
        self._running = False
        self._thread = None

    def _sample_loop(self):
        # Prime cpu_percent (first call always returns 0.0)
        self._process.cpu_percent(interval=None)
        while self._running:
            time.sleep(self.interval)
            try:
                cpu = self._process.cpu_percent(interval=None)
                mem_mb = self._process.memory_info().rss / (1024 * 1024)
                self._samples_cpu.append(cpu)
                self._samples_mem.append(mem_mb)
            except psutil.NoSuchProcess:
                break

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

        if not self._samples_cpu:
            return {
                "cpu_percent_mean": 0.0, "cpu_percent_max": 0.0,
                "memory_mb_mean": 0.0, "memory_mb_max": 0.0,
            }

        return {
            "cpu_percent_mean": round(sum(self._samples_cpu) / len(self._samples_cpu), 2),
            "cpu_percent_max": round(max(self._samples_cpu), 2),
            "memory_mb_mean": round(sum(self._samples_mem) / len(self._samples_mem), 2),
            "memory_mb_max": round(max(self._samples_mem), 2),
        }


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw_dataframe() -> pd.DataFrame:
    true_path = os.path.join(DATA_DIR, TRUE_CSV)
    fake_path = os.path.join(DATA_DIR, FAKE_CSV)

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        raise FileNotFoundError(
            f"Expected {true_path} and {fake_path}. "
            f"Set DATA_DIR to the folder containing them."
        )

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true["label"] = 1  # real
    df_fake["label"] = 0  # fake

    df = pd.concat([df_true, df_fake], ignore_index=True)

    text_cols = [c for c in ["title", "text"] if c in df.columns]
    if not text_cols:
        text_cols = [c for c in df.columns if c != "label"][:1]
    df["content"] = df[text_cols].astype(str).agg(" ".join, axis=1)
    df["content"] = df["content"].apply(_clean_text)

    df = df.dropna(subset=["content"]).reset_index(drop=True)
    return df[["content", "label"]]


def load_train_test_split(test_size: float = 0.2, random_state: int = None):
    if random_state is None:
        random_state = SEED
    df = load_raw_dataframe()
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=test_size,
        random_state=random_state, stratify=df["label"]
    )
    return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()


def vectorize_tfidf(X_train, X_test, max_features: int = 20000):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def save_metrics(name: str, metrics: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metrics = {**metrics, "seed": SEED}
    out_path = os.path.join(OUTPUT_DIR, f"{name}_seed{SEED}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{name}] metrics saved to {out_path}")
    print(json.dumps(metrics, indent=2))


def save_predictions(name: str, y_test, y_proba):
    import numpy as np
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_seed{SEED}_predictions.npz")
    np.savez(out_path, y_test=np.asarray(y_test), y_proba=np.asarray(y_proba))
    print(f"[{name}] predictions saved to {out_path}")