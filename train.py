import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# --- 1. Configuration ---
IMAGE_SIZE = (224, 224) # Standard size for VGG16/most models
BATCH_SIZE = 32
DATA_DIR = 'rvf10k' # Your main dataset folder name
EPOCHS = 15 # You might need more or less depending on convergence

# --- 2. Data Loading and Preprocessing ---
## 2.1. Create Data Generators with Augmentation

# Data Augmentation for Training to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values
    rotation_range=20,        # Randomly rotate images
    width_shift_range=0.2,    # Randomly shift widthheight_shift_range=0.2,  # Randomly shift height
    height_shift_range=0.2,   # Randomly shift height
    horizontal_flip=True,     # Randomly flip images
    fill_mode='nearest'       # Fill in new pixels
)

# Only normalization/rescaling for Validation (no augmentation)
valid_datagen = ImageDataGenerator(rescale=1./255)

## 2.2. Load Data from Directories

train_generator = train_datagen.flow_from_directory(
    directory=f'{DATA_DIR}/train',
    target_size=IMAGE_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='binary',  # For a two-class problem (0 or 1)
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    directory=f'{DATA_DIR}/valid',
    target_size=IMAGE_SIZE,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # No need to shuffle validation data
)

# Print the class mapping: {'fake': 0, 'real': 1} or vice-versa
print("Class Indices:", train_generator.class_indices)

# --- 3. Build the Model (Transfer Learning with VGG16) ---
## 3.1. Load Pre-trained Base Model
base_model = VGG16(
    weights='imagenet',        # Use weights pre-trained on a massive dataset
    include_top=False,         # Exclude the VGG16 final classification layer
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

# Freeze the base model layers (prevent them from being updated during training)
base_model.trainable = False 

## 3.2. Define the Custom Top Layers
model = Sequential([
    base_model,                                     # VGG16 Feature Extractor
    Flatten(),                                      # Flatten the 3D output to 1D
    Dense(512, activation='relu'),                  # Fully connected layer
    Dropout(0.5),                                   # Dropout for regularization
    Dense(1, activation='sigmoid')                  # Output layer for binary classification
])

# --- 4. Compile and Train the Model ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Appropriate loss for two classes
    metrics=['accuracy']
)

model.summary()
# 

# 4.2. Train the Model
print("\nStarting model training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=valid_generator.samples // BATCH_SIZE
)

# --- 5. Save the Model ---
model.save('real_vs_fake_face_model.h5')
print("\nModel saved as 'real_vs_fake_face_model.h5'")