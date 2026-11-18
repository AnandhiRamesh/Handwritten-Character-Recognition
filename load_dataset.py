
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Display dataset information
print("MNIST Dataset Loaded Successfully!")

print("\nTraining Data:")
print("Images:", x_train.shape)
print("Labels:", y_train.shape)

print("\nTesting Data:")
print("Images:", x_test.shape)
print("Labels:", y_test.shape)

print("\nNumber of Classes:", len(np.unique(y_train)))

# Display sample handwritten digits
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
