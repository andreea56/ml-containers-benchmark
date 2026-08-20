import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utile import load_train_test_split, save_metrics, save_predictions, SEED, ResourceMonitor

MAX_VOCAB = 20000
MAX_LEN = 200
EMBED_DIM = 128


def build_model(vocab_size: int) -> Model:
    inputs = layers.Input(shape=(MAX_LEN,), name="tokens")
    x = layers.Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN)(inputs)

    convs = []
    for kernel_size in (3, 4, 5):
        c = layers.Conv1D(filters=100, kernel_size=kernel_size, activation="relu")(x)
        c = layers.GlobalMaxPooling1D()(c)
        convs.append(c)
    x = layers.Concatenate()(convs)

    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="deeptext_cnn")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    start = time.time()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    monitor = ResourceMonitor()
    monitor.start()

    X_train, X_test, y_train, y_test = load_train_test_split()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(
        X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(
        X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post"
    )

    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    model = build_model(vocab_size)

    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)

    epochs = int(os.environ.get("EPOCHS", 3))
    model.fit(
        X_train_pad, y_train_arr,
        validation_split=0.1,
        epochs=epochs,
        batch_size=64,
        verbose=2,
    )

    y_proba = model.predict(X_test_pad, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_arr, y_pred, average="binary"
    )

    resource_stats = monitor.stop()
    elapsed = round(time.time() - start, 3)

    metrics = {
        "algorithm": "DeepText-inspired CNN",
        "accuracy": accuracy_score(y_test_arr, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc_score(y_test_arr, y_proba),
        "vocab_size": vocab_size,
        "epochs": epochs,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_time_seconds": elapsed,
        "throughput_samples_per_second": round(len(X_train) / elapsed, 2),
        **resource_stats,
    }

    save_metrics("deeptext", metrics)
    save_predictions("deeptext", y_test_arr, y_proba)


if __name__ == "__main__":
    main()