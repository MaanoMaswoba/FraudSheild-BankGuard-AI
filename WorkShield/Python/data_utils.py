"""Dataset validation and TensorFlow input-pipeline helpers."""
from pathlib import Path
import tensorflow as tf
from Python.config import (TRAIN_DIR, VALID_DIR, TEST_DIR, CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED, SUPPORTED_IMAGE_EXTENSIONS)

def count_class_images(split_dir):
    split_dir = Path(split_dir)
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            counts[class_name] = 0
        else:
            counts[class_name] = sum(
                1 for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
    return counts

def validate_dataset(require_test=False):
    errors = []
    required = [("training", TRAIN_DIR), ("validation", VALID_DIR)]
    if require_test:
        required.append(("test", TEST_DIR))
    for split_name, split_dir in required:
        if not split_dir.exists():
            errors.append(f"Missing {split_name} directory: {split_dir}")
            continue
        counts = count_class_images(split_dir)
        for class_name in CLASS_NAMES:
            if counts[class_name] == 0:
                errors.append(f"No images for '{class_name}' in {split_name}: {split_dir / class_name}")
    return errors

def make_dataset(directory, shuffle):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {directory}")
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(directory), labels="inferred", label_mode="binary",
        class_names=list(CLASS_NAMES), color_mode="rgb",
        batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
        shuffle=shuffle, seed=RANDOM_SEED
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

def calculate_class_weights():
    counts = count_class_images(TRAIN_DIR)
    total = sum(counts.values())
    if total == 0 or any(counts[name] == 0 for name in CLASS_NAMES):
        raise ValueError("Training data must contain images in both fake and real folders.")
    weights = {i: total / (len(CLASS_NAMES) * counts[name]) for i, name in enumerate(CLASS_NAMES)}
    return weights, counts
