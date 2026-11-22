"""Train the WorkShield VGG16 REAL/FAKE image classifier."""
import json
import platform
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from Python.config import (TRAIN_DIR, VALID_DIR, MODEL_PATH, METADATA_PATH, HISTORY_PATH, TRAINING_PLOT_PATH, IMAGE_SIZE, BATCH_SIZE, EPOCHS, CLASS_TO_INDEX, DEFAULT_THRESHOLD, create_output_directories)
from Python.data_utils import validate_dataset, make_dataset, calculate_class_weights
from Python.model_components import build_workshield_model

def save_training_history(history):
    data = {k: [float(v) for v in values] for k, values in history.history.items()}
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    if "accuracy" in data and "val_accuracy" in data:
        epochs = range(1, len(data["accuracy"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, data["accuracy"], label="Training accuracy")
        axes[0].plot(epochs, data["val_accuracy"], label="Validation accuracy")
        axes[0].set(title="Model Accuracy", xlabel="Epoch", ylabel="Accuracy")
        axes[0].legend()
        axes[1].plot(epochs, data["loss"], label="Training loss")
        axes[1].plot(epochs, data["val_loss"], label="Validation loss")
        axes[1].set(title="Model Loss", xlabel="Epoch", ylabel="Loss")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(TRAINING_PLOT_PATH, dpi=150)
        plt.close(fig)

def main():
    create_output_directories()
    print("=" * 60)
    print("                 WORKSHIELD TRAINING")
    print("=" * 60)
    print(f"Python: {platform.python_version()}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Training folder: {TRAIN_DIR}")
    print(f"Validation folder: {VALID_DIR}")
    print(f"Image size: {IMAGE_SIZE}; batch size: {BATCH_SIZE}; epochs: {EPOCHS}")

    errors = validate_dataset()
    if errors:
        print("\nDATASET VALIDATION FAILED:")
        for error in errors:
            print("-", error)
        raise SystemExit("Add the dataset images to the required folders, then retry.")

    train_ds = make_dataset(TRAIN_DIR, shuffle=True)
    valid_ds = make_dataset(VALID_DIR, shuffle=False)
    class_weights, class_counts = calculate_class_weights()
    print("\nClass mapping:", CLASS_TO_INDEX)
    print("Training image counts:", class_counts)
    print("Class weights:", class_weights)

    model, _ = build_workshield_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")]
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.2, patience=2, min_lr=1e-7, verbose=1)
    ]

    print("\nStarting training...")
    start = time.perf_counter()
    history = model.fit(
        train_ds, validation_data=valid_ds, epochs=EPOCHS,
        class_weight=class_weights, callbacks=callbacks, verbose=1
    )
    duration = time.perf_counter() - start

    # EarlyStopping restores the best validation-loss weights. Save those weights.
    model.save(MODEL_PATH)
    metadata = {
        "project": "WorkShield",
        "model_name": "VGG16 transfer-learning REAL/FAKE classifier",
        "class_indices": CLASS_TO_INDEX,
        "image_size": list(IMAGE_SIZE),
        "threshold": DEFAULT_THRESHOLD,
        "preprocessing": "VGG16 preprocess_input embedded in model",
        "training_images": class_counts,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": EPOCHS,
        "epochs_completed": len(history.history.get("loss", [])),
        "training_seconds": round(duration, 3),
        "base_model": "VGG16 ImageNet weights, frozen",
        "classification_scope": "Static facial image REAL/FAKE classification only"
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    save_training_history(history)

    print("\nTraining completed.")
    print(f"Training time: {duration / 60:.2f} minutes")
    print(f"Model: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print(f"Training graph: {TRAINING_PLOT_PATH}")
    if "val_accuracy" in history.history:
        print(f"Best validation accuracy: {max(history.history['val_accuracy']) * 100:.2f}%")
    print("Next: python -m Python.evaluate_model")

if __name__ == "__main__":
    main()
