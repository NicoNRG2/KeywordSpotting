"""
Keyword Spotting Pipeline - Edge Impulse Dataset
=================================================
Classi target    : heynano, on, off
Background       : noise + unknown -> "_background_"

Esecuzione: python keyword_spotting.py
"""

import os
import json
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
#  CONFIGURAZIONE
# ============================================================
DATASET_DIR   = "./dataset"
LABELS_JSON   = "./dataset/info.labels"
OUTPUT_DIR    = "./saved_model"

SAMPLE_RATE   = 16000
DURATION      = 1.0
N_MFCC        = 13
HOP_LENGTH    = 320
N_FFT         = 512

EPOCHS        = 50
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3

TARGET_LABELS = ["heynano", "on", "off"]
CLASS_NAMES   = TARGET_LABELS + ["_background_"]
NUM_CLASSES   = len(CLASS_NAMES)
LABEL2IDX     = {name: i for i, name in enumerate(CLASS_NAMES)}


# ============================================================
#  1. ORGANIZZAZIONE DATASET
# ============================================================

def load_file_manifest():
    print("[1/5] Caricamento manifest ->", LABELS_JSON)
    with open(LABELS_JSON, "r") as f:
        data = json.load(f)

    train_files, test_files = [], []
    for entry in data["files"]:
        raw_label = entry["label"]["label"]
        label = raw_label if raw_label in TARGET_LABELS else "_background_"
        record = {
            "path":  os.path.join(DATASET_DIR, entry["path"]),
            "label": label,
        }
        if entry["category"] == "training":
            train_files.append(record)
        else:
            test_files.append(record)

    from collections import Counter
    print("      Training:", dict(Counter(f["label"] for f in train_files)))
    print("      Testing :", dict(Counter(f["label"] for f in test_files)))
    return train_files, test_files


# ============================================================
#  2. FEATURE EXTRACTION - MFCC
# ============================================================

def load_audio(file_path):
    target_len = int(SAMPLE_RATE * DURATION)
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    except Exception as e:
        print("\n      WARN: impossibile leggere", file_path, "-", e)
        return np.zeros(target_len, dtype=np.float32)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio.astype(np.float32)


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=32,  
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
    )
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
    return mfcc.T  # (time_frames, N_MFCC)


def build_features(file_list, split_name):
    X, y = [], []
    total = len(file_list)
    for i, entry in enumerate(file_list):
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"      {split_name}: {i+1}/{total}", end="\r")
        audio = load_audio(entry["path"])
        X.append(extract_mfcc(audio))
        y.append(LABEL2IDX[entry["label"]])
    print()
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ============================================================
#  3. MODELLO Conv1D
# ============================================================

def build_model(input_shape):
    inputs = keras.Input(shape=input_shape, name="mfcc_input")

    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    return keras.Model(inputs, outputs, name="KeywordSpotting_Conv1D")


# ============================================================
#  4. TRAINING
# ============================================================

def train_model(X_train, y_train, X_test, y_test):
    print("\n[4/5] Training")
    print("      Input shape :", X_train.shape[1:])
    print("      Classi      :", CLASS_NAMES)
    print("      Train/Test  :", len(X_train), "/", len(X_test))

    model = build_model(X_train.shape[1:])
    model.summary()

    weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train,
    )
    class_weights = {i: float(w) for i, w in enumerate(weights_arr)}
    print("      Class weights:", {CLASS_NAMES[i]: round(w, 2) for i, w in class_weights.items()})

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, "best_checkpoint.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


# ============================================================
#  5. VALUTAZIONE E SALVATAGGIO
# ============================================================

def evaluate_and_save(model, history, X_test, y_test):
    print("\n[5/5] Valutazione e salvataggio in:", OUTPUT_DIR)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print("      Test Loss    :", round(loss, 4))
    print("      Test Accuracy:", round(acc, 4))

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],     label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(history.history["loss"],     label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
    plt.close()

    model_path = os.path.join(OUTPUT_DIR, "keyword_spotting_model.keras")
    model.save(model_path)

    meta = {
        "class_names": CLASS_NAMES,
        "label2idx":   LABEL2IDX,
        "sample_rate": SAMPLE_RATE,
        "duration":    DURATION,
        "n_mfcc":      N_MFCC,
        "hop_length":  HOP_LENGTH,
        "n_fft":       N_FFT,
        "input_shape": list(model.input_shape[1:]),
    }
    with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone!")
    print("  ->", model_path)
    print("  ->", os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    print("  ->", os.path.join(OUTPUT_DIR, "training_history.png"))
    print("  ->", os.path.join(OUTPUT_DIR, "model_metadata.json"))


# ============================================================
#  INFERENCE HELPER (uso post-training)
# ============================================================

def predict_file(wav_path):
    """
    Predice la classe di un singolo file .wav usando il modello salvato.

    Esempio:
        from keyword_spotting import predict_file
        predict_file("mio_audio.wav")
    """
    model = keras.models.load_model(os.path.join(OUTPUT_DIR, "keyword_spotting_model.keras"))
    audio = load_audio(wav_path)
    mfcc  = extract_mfcc(audio)
    probs = model.predict(mfcc[np.newaxis, ...], verbose=0)[0]
    pred_label = CLASS_NAMES[np.argmax(probs)]
    confidence = float(probs.max())
    print("File     :", wav_path)
    print(f"Predicted: {pred_label} ({confidence:.2%})")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name:15s}: {p:.4f}")
    return pred_label, confidence


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_files, test_files = load_file_manifest()

    print(f"\n[3/5] Feature extraction MFCC (n_mfcc={N_MFCC})")
    X_train, y_train = build_features(train_files, "Training")
    X_test,  y_test  = build_features(test_files,  "Testing ")
    print("      X_train:", X_train.shape, " | X_test:", X_test.shape)

    model, history = train_model(X_train, y_train, X_test, y_test)

    evaluate_and_save(model, history, X_test, y_test)