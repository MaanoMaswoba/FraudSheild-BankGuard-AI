"""Reusable WorkShield model components."""
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from Python.config import IMAGE_SIZE

@tf.keras.utils.register_keras_serializable(package="WorkShield")
class VGG16Preprocessing(layers.Layer):
    """Apply official VGG16 preprocessing inside the saved model."""
    def call(self, inputs):
        return tf.keras.applications.vgg16.preprocess_input(
            tf.cast(inputs, tf.float32)
        )

def build_workshield_model():
    """Build a frozen-VGG16 transfer-learning REAL/FAKE classifier."""
    inputs = layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        name="facial_image"
    )
    x = layers.RandomFlip("horizontal", name="horizontal_flip")(inputs)
    x = layers.RandomRotation(0.03, fill_mode="nearest", name="random_rotation")(x)
    x = layers.RandomTranslation(0.05, 0.05, fill_mode="nearest", name="random_translation")(x)
    x = layers.RandomZoom(0.08, fill_mode="nearest", name="random_zoom")(x)
    x = VGG16Preprocessing(name="vgg16_preprocessing")(x)

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(0.0001),
        name="classification_features"
    )(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="real_probability")(x)
    return Model(inputs, outputs, name="WorkShield_VGG16"), base_model
