import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize the images to [0, 1] range and reshape them to 28x28x1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Define the VAE model

latent_dim = 5  # You can experiment with different latent dimensions

# Encoder
inputs = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(inputs)
x = layers.Dense(128, activation='relu')(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Sampling layer: Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(128, activation='relu')(latent_inputs)
x = layers.Dense(28 * 28, activation='sigmoid')(x)
outputs = layers.Reshape((28, 28, 1))(x)

# Instantiate the Encoder and Decoder models
encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = models.Model(latent_inputs, outputs, name='decoder')

# Define the VAE model that combines the encoder and decoder
vae_outputs = decoder(encoder(inputs)[2])
vae = models.Model(inputs, vae_outputs, name='vae')

# Loss function (VAE loss = reconstruction loss + KL divergence)
reconstruction_loss = tf.reduce_mean(
    tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(inputs, vae_outputs), axis=(1, 2)
    )
)
kl_loss = -0.5 * tf.reduce_mean(
    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
)
vae_loss = reconstruction_loss + kl_loss

# Compile the VAE model
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train the VAE model
vae.fit(x_train, x_train, epochs=30, batch_size=128, validation_data=(x_test, x_test))

# Sampling from the latent space to generate new images
def generate_images(latent_points):
    generated_images = decoder.predict(latent_points)
    return generated_images

# Generate new images by sampling random points from the latent space
n = 20  # Number of images to generate
figure, axes = plt.subplots(1, n, figsize=(20, 4))
for i in range(n):
    random_latent_vector = np.random.normal(size=(1, latent_dim))
    generated_image = generate_images(random_latent_vector)
    axes[i].imshow(generated_image[0, :, :, 0], cmap='gray')
    axes[i].axis('off')

plt.show()



