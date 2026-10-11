"""
Reproducible Model Evaluation and Benchmark Script for Indian Sign Language Recognition.
Strictly evaluates model.h5 on a held-out test split of keypoint.csv without data leakage.
Distinguishes between validated classes (A, B, C) and unvalidated architecture classes.
Measures isolated neural inference latency and exports an honest evaluation_report.json.
"""

import os
import time
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from core.model_engine import ISLModelEngine


def evaluate_isl_model(
    data_path: str = "keypoint.csv",
    model_path: str = "model.h5",
    test_size: float = 0.20,
    random_seed: int = 42,
    latency_iterations: int = 1000
) -> Dict[str, Any]:
    print("==================================================================")
    print("  INDIAN SIGN LANGUAGE MODEL EVALUATION & BENCHMARK SUITE")
    print("==================================================================")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # 1. Dataset Inspection
    print(f"\n[1/5] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path, header=None)
    total_samples, total_cols = df.shape
    features_count = total_cols - 1

    label_counts = df[0].value_counts().to_dict()
    represented_classes = sorted(list(label_counts.keys()))

    print(f"      Total Samples: {total_samples}")
    print(f"      Feature Dimensions: {features_count}")
    print(f"      Represented Classes in Dataset: {represented_classes}")
    for cls_name, cnt in label_counts.items():
        print(f"        Class '{cls_name}': {cnt} samples ({cnt / total_samples * 100:.1f}%)")

    # 2. Strict Stratified Train/Test Split
    print(f"\n[2/5] Creating strict stratified held-out test split ({int(test_size*100)}% test, seed={random_seed})...")
    X = df.iloc[:, 1:].values.astype(np.float32)
    y = df.iloc[:, 0].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    print(f"      Training Set: {len(y_train)} samples (isolated)")
    print(f"      Held-Out Test Set: {len(y_test)} samples (used strictly for evaluation)")

    # 3. Model Engine Initialization
    print(f"\n[3/5] Loading neural model from '{model_path}'...")
    engine = ISLModelEngine(model_path)
    all_supported_classes = engine.get_supported_classes()
    unvalidated_classes = [c for c in all_supported_classes if c not in represented_classes]

    print(f"      Model Input Dimensions: {engine.INPUT_DIM}")
    print(f"      Model Output Classes: {engine.NUM_CLASSES}")
    print(f"      Classes with Ground-Truth Data: {represented_classes} ({len(represented_classes)})")
    print(f"      Classes Lacking Ground-Truth Data in keypoint.csv: {len(unvalidated_classes)} classes")

    # 4. Latency Benchmark (Isolated from I/O and Preprocessing)
    print(f"\n[4/5] Measuring pure neural inference latency ({latency_iterations} iterations)...")
    sample_feature = X_test[0:1]  # Shape (1, 42)

    # Warm-up pass
    for _ in range(50):
        _ = engine.forward(sample_feature)

    latencies_ms: List[float] = []
    for _ in range(latency_iterations):
        t0 = time.perf_counter()
        _ = engine.forward(sample_feature)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_latency = float(np.mean(latencies_ms))
    median_latency = float(np.median(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))
    throughput_fps = float(1000.0 / mean_latency) if mean_latency > 0 else 0.0

    print(f"      Mean Latency:   {mean_latency:.4f} ms")
    print(f"      Median (P50):   {median_latency:.4f} ms")
    print(f"      95th Pct (P95): {p95_latency:.4f} ms")
    print(f"      99th Pct (P99): {p99_latency:.4f} ms")
    print(f"      Pure Throughput: {throughput_fps:.1f} inferences/sec")

    # 5. Model Inference & Metrics Calculation
    print("\n[5/5] Running inference on held-out test split...")
    test_probs = engine.forward(X_test)
    pred_indices = np.argmax(test_probs, axis=1)
    y_pred = [all_supported_classes[idx] for idx in pred_indices]

    # Overall accuracy
    accuracy = float(accuracy_score(y_test, y_pred))

    # Metrics restricted to the represented test labels
    precision_macro = float(precision_score(y_test, y_pred, labels=represented_classes, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, labels=represented_classes, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, labels=represented_classes, average="macro", zero_division=0))

    precision_weighted = float(precision_score(y_test, y_pred, labels=represented_classes, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, labels=represented_classes, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, labels=represented_classes, average="weighted", zero_division=0))

    class_report_dict = classification_report(
        y_test, y_pred, labels=represented_classes, target_names=represented_classes,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=represented_classes).tolist()

    # Out-of-represented predictions check
    out_of_dist_preds = [p for p in y_pred if p not in represented_classes]
    ood_counts = {str(k): int(v) for k, v in pd.Series(out_of_dist_preds).value_counts().items()} if out_of_dist_preds else {}
    label_counts_clean = {str(k): int(v) for k, v in label_counts.items()}

    print("\n======================= EVALUATION RESULTS =======================")
    print(f"  Validated Classes:      {represented_classes}")
    print(f"  Test Samples Evaluated: {len(y_test)}")
    print(f"  Test Accuracy:          {accuracy * 100:.2f}%")
    print(f"  Macro Precision:        {precision_macro:.4f}")
    print(f"  Macro Recall:           {recall_macro:.4f}")
    print(f"  Macro F1-Score:         {f1_macro:.4f}")
    print(f"  Weighted F1-Score:      {f1_weighted:.4f}")
    if ood_counts:
        print(f"  False Predictions outside ground-truth: {ood_counts}")
    print("------------------------------------------------------------------")
    print("Per-Class Metrics on Held-Out Test Split:")
    print(classification_report(y_test, y_pred, labels=represented_classes, digits=4, zero_division=0))
    print("Confusion Matrix (Rows=True, Cols=Predicted):")
    print(f"        {' '.join([f'{c:>6}' for c in represented_classes])}")
    for idx, row in enumerate(cm):
        print(f"  {represented_classes[idx]:>4}: {' '.join([f'{val:>6}' for val in row])}")
    print("==================================================================")

    # Compile honest, verifiable report
    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "source": data_path,
            "total_samples": total_samples,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "test_split_ratio": test_size,
            "random_seed": random_seed,
            "feature_dimension": features_count,
            "validated_classes": represented_classes,
            "class_distribution": label_counts_clean,
            "data_limitation_statement": (
                "The repository's keypoint.csv currently contains pre-computed landmark samples "
                "for classes 'A', 'B', and 'C'. Therefore, empirical test-set metrics can only be "
                "statistically validated on these 3 classes."
            )
        },
        "model": {
            "path": model_path,
            "architecture": "Deep Feedforward Neural Network (5 Dense Layers + Dropout)",
            "input_dimension": engine.INPUT_DIM,
            "output_neurons": engine.NUM_CLASSES,
            "all_output_classes": all_supported_classes,
            "unvalidated_classes_count": len(unvalidated_classes),
            "unvalidated_classes": unvalidated_classes
        },
        "metrics_on_held_out_test_set": {
            "test_accuracy": round(accuracy, 4),
            "macro_precision": round(precision_macro, 4),
            "macro_recall": round(recall_macro, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_precision": round(precision_weighted, 4),
            "weighted_recall": round(recall_weighted, 4),
            "weighted_f1": round(f1_weighted, 4)
        },
        "per_class_performance": class_report_dict,
        "confusion_matrix": {
            "labels": represented_classes,
            "matrix": cm
        },
        "false_predictions_outside_ground_truth": ood_counts,
        "latency_benchmark_ms": {
            "mean_ms": round(mean_latency, 4),
            "median_p50_ms": round(median_latency, 4),
            "p95_ms": round(p95_latency, 4),
            "p99_ms": round(p99_latency, 4),
            "throughput_inferences_per_sec": round(throughput_fps, 1),
            "measured_iterations": latency_iterations
        }
    }

    report_path = "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[INFO] Validated evaluation report exported to '{report_path}'.")
    return report


if __name__ == "__main__":
    evaluate_isl_model()
