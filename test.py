import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# --- Configuration ---
# Match this to the size used during training
IMAGE_SIZE = (224, 224) 
MODEL_PATH = 'real_vs_fake_face_model.h5'

# --- The image you want to test ---
# Replace this with the actual path to an image you want to test.
# Example: 'rvf10k/valid/fake/0AS16QER71.jpg' 
TEST_IMAGE_PATH = 'rvf10k/valid/fake/0AIFZB4IE6.jpg' # <--- CHANGE THIS PATH

# Map the indices back to human-readable labels
# IMPORTANT: Check your train.py output for the correct mapping.
# If 'fake' was 0 and 'real' was 1, use this order:
CLASS_LABELS = {0: 'FAKE', 1: 'REAL'} 

# --- 1. Load the Trained Model ---
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}. Run train.py first!")
    exit()

try:
    # Load the model structure and weights
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 2. Preprocess the Image ---
def load_and_preprocess_image(img_path, target_size):
    """Loads an image, resizes it, and converts it to a ready-to-use tensor."""
    if not os.path.exists(img_path):
        print(f"ERROR: Test image not found at {img_path}")
        return None

    # Load the image using Keras utility
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert image to numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to create a batch of size 1 (required by the model)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale/Normalize pixel values (matching the 1./255. used in training)
    img_array /= 255.0
    
    return img_array

# Load the test image data
test_image_tensor = load_and_preprocess_image(TEST_IMAGE_PATH, IMAGE_SIZE)

if test_image_tensor is None:
    exit()

# --- 3. Make Prediction ---
print("\nPredicting...")
prediction = model.predict(test_image_tensor)

# The output is a probability (float) between 0 and 1
probability_of_fake = prediction[0][0]

# --- 4. Interpret Result ---
# If probability is < 0.5, it's classified as the 0-index (e.g., FAKE).
# If probability is >= 0.5, it's classified as the 1-index (e.g., REAL).
predicted_class_index = int(probability_of_fake >= 0.5)
predicted_label = CLASS_LABELS.get(predicted_class_index, "UNKNOWN")

# Calculate probability percentages for better insight
prob_real = probability_of_fake * 100
prob_fake = (1 - probability_of_fake) * 100

print("-" * 50)
print(f"Test Image: {os.path.basename(TEST_IMAGE_PATH)}")
print(f"-> Prediction: **{predicted_label}**")
print(f"-> Confidence: {prob_fake:.2f}% FAKE / {prob_real:.2f}% REAL")
print("-" * 50)