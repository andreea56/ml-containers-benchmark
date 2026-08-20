import os
import time
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utile import load_train_test_split, vectorize_tfidf, save_metrics, save_predictions, SEED, ResourceMonitor

SVD_DIM = int(os.environ.get("SVD_DIM", 100))
GAMMA = float(os.environ.get("GAMMA", 0.0))  # single-step episodes: no future reward
LR = float(os.environ.get("LR", 1e-3))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
BUFFER_SIZE = int(os.environ.get("BUFFER_SIZE", 20000))
EPSILON_START = float(os.environ.get("EPSILON_START", 1.0))
EPSILON_MIN = float(os.environ.get("EPSILON_MIN", 0.01))
EPSILON_DECAY = float(os.environ.get("EPSILON_DECAY", 0.999))
N_EPOCHS = int(os.environ.get("N_EPOCHS", 2))


def build_q_network(state_dim: int) -> Model:
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(2, activation="linear")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
    return model


def main():
    start = time.time()
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    monitor = ResourceMonitor()
    monitor.start()

    X_train, X_test, y_train, y_test = load_train_test_split()
    X_train_vec, X_test_vec, _ = vectorize_tfidf(X_train, X_test)

    svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
    X_train_reduced = svd.fit_transform(X_train_vec).astype(np.float32)
    X_test_reduced = svd.transform(X_test_vec).astype(np.float32)

    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)

    q_network = build_q_network(SVD_DIM)
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epsilon = EPSILON_START
    n_train = len(y_train_arr)
    order_rng = np.random.RandomState(SEED)

    for epoch in range(N_EPOCHS):
        order = order_rng.permutation(n_train)
        for step, idx in enumerate(order):
            state = X_train_reduced[idx]
            true_label = y_train_arr[idx]

            if np.random.random() < epsilon:
                action = np.random.randint(2)
            else:
                q_values = q_network.predict(state[np.newaxis, :], verbose=0)
                action = int(np.argmax(q_values[0]))

            reward = 1.0 if action == true_label else -1.0
            replay_buffer.append((state, action, reward))
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

            if len(replay_buffer) >= BATCH_SIZE and step % 4 == 0:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                states = np.array([b[0] for b in batch])
                actions = np.array([b[1] for b in batch])
                rewards = np.array([b[2] for b in batch])

                target_q = q_network.predict(states, verbose=0)
                for i in range(BATCH_SIZE):
                    target_q[i, actions[i]] = rewards[i]  # GAMMA=0: no next state

                q_network.fit(states, target_q, epochs=1, verbose=0)

    q_test = q_network.predict(X_test_reduced, verbose=0)
    y_pred = np.argmax(q_test, axis=1)

    exp_q = np.exp(q_test - q_test.max(axis=1, keepdims=True))
    probs = exp_q / exp_q.sum(axis=1, keepdims=True)
    y_proba = probs[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_arr, y_pred, average="binary"
    )

    resource_stats = monitor.stop()
    elapsed = round(time.time() - start, 3)

    metrics = {
        "algorithm": "DQN-bandit (on news text)",
        "svd_dim": SVD_DIM,
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
    save_metrics("dqn_bandit", metrics)
    save_predictions("dqn_bandit", y_test_arr, y_proba)


if __name__ == "__main__":
    main()
