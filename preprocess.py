
import tensorflow as tf
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Before Preprocessing:")
print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)
print("Pixel value range:", x_train.min(), "to", x_train.max())

# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension for CNN
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("\nAfter Preprocessing:")
print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)
print("Pixel value range:", x_train.min(), "to", x_train.max())

print("\nData preprocessing completed successfully!")
