# app.py
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
import io

# Define the Vector Quantizer layer and VQ-VAE model components
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.embeddings = tf.Variable(
            initial_value=tf.random.uniform(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        return tf.argmin(distances, axis=1)

# Define the Encoder, Decoder, and VQ-VAE models
def get_encoder(latent_dim=16):
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")

def get_decoder(latent_dim=16):
    latent_inputs = keras.Input(shape=(7, 7, latent_dim))
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(latent_inputs)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

def get_vqvae(latent_dim=16, num_embeddings=128):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(28, 28, 1))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")

# Instantiate and load the model
latent_dim = 16
num_embeddings = 128
vqvae_model = get_vqvae(latent_dim, num_embeddings)
vqvae_model.load_weights("vqvae_mnist.weights.h5")  # Ensure this file is uploaded to the repo

# Gradio utility functions
def process_input(input_image):
    if len(input_image.shape) == 3:
        input_image = np.mean(input_image, axis=-1)
    input_image = tf.image.resize(input_image[np.newaxis, ..., np.newaxis], (28, 28)).numpy().squeeze()
    return (input_image / 255.0) - 0.5

def predict(image):
    processed_image = process_input(image)
    input_tensor = processed_image.reshape(1, 28, 28, 1)
    reconstruction = vqvae_model.predict(input_tensor)
    comparison_img = show_subplot(processed_image, reconstruction[0])
    return comparison_img

def show_subplot(original, reconstructed):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze() + 0.5, cmap='gray')
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze() + 0.5, cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Image(type="pil"),
    title="VQ-VAE Image Reconstruction",
    description="Upload an image to see the original and reconstructed versions side by side"
)

iface.launch()

