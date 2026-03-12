"""
Evaluate Keyword Spotting Model on Testing Folder
=================================================

Esegue inferenza su tutti i sample del set "testing"
e stampa le metriche principali.

Usage:
    python evaluate_testing_folder.py
"""

import os
import json
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# importa funzioni dal tuo script
from main import load_audio, extract_mfcc

# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = "./dataset"
LABELS_JSON = "./dataset/info.labels"
MODEL_PATH  = "./saved_model/keyword_spotting_model.keras"

TARGET_LABELS = ["heynano", "on", "off"]
CLASS_NAMES   = TARGET_LABELS + ["_background_"]
LABEL2IDX     = {name: i for i, name in enumerate(CLASS_NAMES)}


# ============================================================
# LOAD TEST FILES
# ============================================================

def load_testing_manifest():
    print("Loading testing manifest...")

    with open(LABELS_JSON, "r") as f:
        data = json.load(f)

    files = []

    for entry in data["files"]:
        if entry["category"] != "testing":
            continue

        raw_label = entry["label"]["label"]
        label = raw_label if raw_label in TARGET_LABELS else "_background_"

        files.append({
            "path": os.path.join(DATASET_DIR, entry["path"]),
            "label": label
        })

    print("Testing samples:", len(files))
    return files


# ============================================================
# INFERENCE
# ============================================================

def run_inference(files):

    print("\nLoading model:", MODEL_PATH)
    model = keras.models.load_model(MODEL_PATH)

    y_true = []
    y_pred = []

    total = len(files)

    for i, sample in enumerate(files):

        if (i+1) % 100 == 0 or (i+1) == total:
            print(f"Inference {i+1}/{total}", end="\r")

        audio = load_audio(sample["path"])
        mfcc  = extract_mfcc(audio)

        probs = model.predict(mfcc[np.newaxis,...], verbose=0)[0]
        pred  = np.argmax(probs)

        y_true.append(LABEL2IDX[sample["label"]])
        y_pred.append(pred)

    print()

    return np.array(y_true), np.array(y_pred)


# ============================================================
# METRICS
# ============================================================

def print_metrics(y_true, y_pred):

    print("\n==============================")
    print("RESULTS")
    print("==============================")

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", round(acc,4))

    print("\nClassification Report\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_files = load_testing_manifest()

    y_true, y_pred = run_inference(test_files)

    print_metrics(y_true, y_pred)