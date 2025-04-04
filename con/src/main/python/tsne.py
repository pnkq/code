import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data: Flatten the images and normalize them
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images (28x28 to 784-dimensional vectors)
x_train_flattened = x_train.reshape((-1, 28 * 28))
x_test_flattened = x_test.reshape((-1, 28 * 28))

# Apply t-SNE for dimensionality reduction to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
x_train_tsne = tsne.fit_transform(x_train_flattened)

# Create a scatter plot of the 2D representation
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], c=y_train, cmap='tab10', s=5, alpha=0.5)
plt.colorbar(scatter)
plt.title("t-SNE Visualization of MNIST (2D)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

