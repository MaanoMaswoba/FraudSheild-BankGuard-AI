"""Shared WorkShield image-prediction functions."""
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from Python.model_components import VGG16Preprocessing  # registers custom layer
from Python.config import MODEL_PATH, METADATA_PATH, IMAGE_SIZE, DEFAULT_THRESHOLD

def load_workshield_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}. Run 'python -m Python.train' first.")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}. Run training again.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model, metadata

def prepare_image(image_path):
    """Load RGB image. VGG16 preprocessing is embedded in the saved model."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = load_img(str(image_path), target_size=IMAGE_SIZE, color_mode="rgb")
    image_array = img_to_array(img).astype("float32")
    return np.expand_dims(image_array, axis=0)

def predict_loaded_model(model, image_path, metadata):
    tensor = prepare_image(image_path)
    threshold = float(metadata.get("threshold", DEFAULT_THRESHOLD))
    start = time.perf_counter()
    output = model.predict(tensor, verbose=0)
    elapsed = time.perf_counter() - start
    real_score = float(np.clip(output[0][0], 0.0, 1.0))
    fake_score = 1.0 - real_score
    predicted_class = "REAL" if real_score >= threshold else "FAKE"
    selected_score = real_score if predicted_class == "REAL" else fake_score
    if abs(real_score - threshold) <= 0.10:
        note = "The result is close to the decision threshold. Human review is recommended."
    elif predicted_class == "FAKE":
        note = "The image was classified as potentially fake. Refer it for further human assessment."
    else:
        note = "The image was classified as REAL by the model. This does not prove authenticity or verify identity."
    return {
        "predicted_class": predicted_class,
        "real_score": real_score,
        "fake_score": fake_score,
        "selected_score": selected_score,
        "threshold": threshold,
        "inference_seconds": elapsed,
        "review_note": note,
        "image_path": str(Path(image_path).resolve())
    }

def predict_image(image_path):
    model, metadata = load_workshield_model()
    return predict_loaded_model(model, image_path, metadata)
