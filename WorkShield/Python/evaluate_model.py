"""Evaluate WorkShield on an independent test split or validation split."""
import argparse
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from Python.config import TEST_DIR, VALID_DIR, INDEX_TO_CLASS, EVALUATION_JSON_PATH, EVALUATION_REPORT_PATH, CONFUSION_MATRIX_PATH, MODEL_PATH, create_output_directories
from Python.data_utils import make_dataset, count_class_images
from Python.inference import load_workshield_model

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WorkShield classification performance.")
    parser.add_argument("--split", choices=["auto", "test", "valid"], default="auto")
    return parser.parse_args()

def has_complete_test_split():
    if not TEST_DIR.exists():
        return False
    counts = count_class_images(TEST_DIR)
    return all(counts.get(name, 0) > 0 for name in ("fake", "real"))


def choose_split(requested):
    if requested == "test":
        if not has_complete_test_split():
            raise FileNotFoundError(
                "A complete independent test split was not found. "
                "Both Data/rvf10k/test/fake and test/real must contain images."
            )
        return TEST_DIR, "INDEPENDENT TEST"
    if requested == "valid":
        if not VALID_DIR.exists():
            raise FileNotFoundError("Validation folder not found.")
        return VALID_DIR, "VALIDATION (NOT AN INDEPENDENT TEST)"
    if has_complete_test_split():
        return TEST_DIR, "INDEPENDENT TEST"
    return VALID_DIR, "VALIDATION (NOT AN INDEPENDENT TEST)"

def save_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix)
    labels = [INDEX_TO_CLASS[0], INDEX_TO_CLASS[1]]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
    ax.set_title("WorkShield Confusion Matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(CONFUSION_MATRIX_PATH, dpi=160); plt.close(fig)

def main():
    create_output_directories()
    args = parse_args()
    evaluation_dir, split_label = choose_split(args.split)
    if not MODEL_PATH.exists():
        raise FileNotFoundError("WorkShield model not found. Run 'python -m Python.train' first.")
    print("=" * 60); print("              WORKSHIELD MODEL EVALUATION"); print("=" * 60)
    print(f"Evaluation source: {split_label}")
    model, metadata = load_workshield_model()
    dataset = make_dataset(evaluation_dir, shuffle=False)
    y_true = np.concatenate([labels.numpy().reshape(-1) for _, labels in dataset]).astype(int)
    process = psutil.Process()
    memory_before = process.memory_info().rss / (1024 * 1024)
    start = time.perf_counter()
    real_scores = model.predict(dataset, verbose=1).reshape(-1)
    elapsed = time.perf_counter() - start
    memory_after = process.memory_info().rss / (1024 * 1024)
    threshold = float(metadata.get("threshold", 0.5))
    y_pred = (real_scores >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, labels=[0, 1], target_names=["FAKE", "REAL"], output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=[0, 1], target_names=["FAKE", "REAL"], zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    auc = float(roc_auc_score(y_true, real_scores)) if len(np.unique(y_true)) == 2 else None
    fake_as_real = int(matrix[0, 1])
    real_as_fake = int(matrix[1, 0])
    average_time = elapsed / len(y_true) if len(y_true) else 0.0
    model_size = MODEL_PATH.stat().st_size / (1024 * 1024)
    results = {
        "evaluation_split": split_label, "evaluation_directory": str(evaluation_dir),
        "number_of_images": int(len(y_true)), "threshold": threshold,
        "accuracy": float(accuracy), "roc_auc_real_as_positive": auc,
        "classification_report": report_dict, "confusion_matrix": matrix.tolist(),
        "fake_images_predicted_as_real": fake_as_real,
        "real_images_predicted_as_fake": real_as_fake,
        "total_prediction_seconds": float(elapsed),
        "average_seconds_per_image": float(average_time),
        "process_memory_before_mb": float(memory_before),
        "process_memory_after_mb": float(memory_after),
        "model_size_mb": float(model_size)
    }
    with open(EVALUATION_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    lines = [
        "WORKSHIELD MODEL EVALUATION", "=" * 45,
        f"Evaluation split: {split_label}", f"Number of images: {len(y_true)}",
        f"Decision threshold: {threshold:.2f}", f"Accuracy: {accuracy * 100:.2f}%",
        f"ROC-AUC (REAL as positive): {auc if auc is not None else 'Not available'}",
        f"Total prediction time: {elapsed:.4f} seconds",
        f"Average time per image: {average_time:.6f} seconds",
        f"Model file size: {model_size:.2f} MB", "",
        "CLASSIFICATION REPORT", report_text, "", "CONFUSION MATRIX", str(matrix), "",
        f"FAKE images incorrectly classified as REAL: {fake_as_real}",
        f"REAL images incorrectly classified as FAKE: {real_as_fake}", "",
        "A REAL prediction is not proof of authenticity or identity.",
        "A FAKE prediction is not proof of fraud or wrongdoing.",
        "Human review and additional verification remain necessary."
    ]
    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    save_confusion_matrix(matrix)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    if auc is not None: print(f"ROC-AUC: {auc:.4f}")
    print("\nClassification report:\n", report_text)
    print("Confusion matrix:\n", matrix)
    print(f"FAKE images predicted REAL: {fake_as_real}")
    print(f"REAL images predicted FAKE: {real_as_fake}")
    print(f"Average prediction time: {average_time:.6f}s/image")
    print(f"Model size: {model_size:.2f} MB")
    if "NOT AN INDEPENDENT TEST" in split_label:
        print("\nWARNING: These are validation results, not independent test results.")
    print("\nSaved evaluation JSON, report and confusion matrix in Results/.")

if __name__ == "__main__":
    main()
