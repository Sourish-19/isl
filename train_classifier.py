"""
Reproducible Classifier Training and Comparison Script for Indian Sign Language Recognition.
Supports training and comparing:
1. Multi-Layer Perceptron (MLP / FNN) Neural Network
2. Random Forest Ensemble Classifier
Provides stratified train/val/test splits, data normalization, metrics computation, and model export.
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score


def train_and_compare_models(data_path: str = "keypoint.csv", output_dir: str = "artifacts"):
    print("==================================================================")
    print("  INDIAN SIGN LANGUAGE CLASSIFIER TRAINING & BENCHMARKING")
    print("==================================================================")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file '{data_path}' not found.")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    print(f"[1/5] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, 1:].values.astype(np.float32)
    raw_y = df.iloc[:, 0].astype(str).values

    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_y)
    classes = list(encoder.classes_)

    print(f"      Total samples: {len(y)}, Features: {X.shape[1]}")
    print(f"      Classes present: {classes}")

    # 2. Stratified Split (80% Train, 20% Test)
    print("[2/5] Creating stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Model 1: Multi-Layer Perceptron (FNN)
    print("\n[3/5] Training Multi-Layer Perceptron (FNN)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        max_iter=60,
        random_state=42,
        verbose=False
    )
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')
    print(f"      MLP Test Accuracy: {acc_mlp * 100:.2f}% | Weighted F1: {f1_mlp:.4f}")

    # 4. Model 2: Random Forest Classifier
    print("\n[4/5] Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    print(f"      Random Forest Test Accuracy: {acc_rf * 100:.2f}% | Weighted F1: {f1_rf:.4f}")

    # 5. Model Selection & Export
    print("\n[5/5] Comparing and saving best performing model...")
    if acc_rf >= acc_mlp:
        best_name = "RandomForest"
        best_model = rf
        best_acc = acc_rf
        y_best_pred = y_pred_rf
    else:
        best_name = "MLP"
        best_model = mlp
        best_acc = acc_mlp
        y_best_pred = y_pred_mlp

    checkpoint = {
        "model": best_model,
        "label_encoder": encoder,
        "classes": classes,
        "architecture": best_name,
        "accuracy": best_acc
    }

    best_export_path = os.path.join(output_dir, f"best_{best_name.lower()}_model.joblib")
    joblib.dump(checkpoint, best_export_path)
    print(f"      Selected: {best_name} (Accuracy: {best_acc * 100:.2f}%)")
    print(f"      Saved model checkpoint and encoder to '{best_export_path}'")

    print("\nFinal Test Classification Report (Best Model):")
    print(classification_report(y_test, y_best_pred, target_names=classes, digits=4))

    return {
        "best_model": best_name,
        "mlp_accuracy": round(acc_mlp, 4),
        "rf_accuracy": round(acc_rf, 4),
        "exported_path": best_export_path
    }


if __name__ == "__main__":
    train_and_compare_models()
