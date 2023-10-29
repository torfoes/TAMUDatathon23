# IMPORTANT
# unless you're willing to change the run.py script, keep the new_case, guess, and add_score methods.

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import layers, Model


# Helper function to convert strokes to image format
def strokes_to_image(strokes):
    image = np.zeros((256, 256), dtype=np.uint8)
    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            cv2.line(image, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), 255, 3)
    return image


# Encoder
class Encoder(Model):
    def __init__(self, enc_rnn_size=256, z_size=128):
        super(Encoder, self).__init__()
        self.rnn = layers.LSTM(enc_rnn_size, return_sequences=False)
        self.fc_mu = layers.Dense(z_size)
        self.fc_logvar = layers.Dense(z_size)

    def call(self, x):
        h = self.rnn(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# Decoder
class Decoder(Model):
    def __init__(self, dec_rnn_size=512, num_mixture=20, z_size=128):
        super(Decoder, self).__init__()
        self.rnn = layers.LSTM(dec_rnn_size, return_sequences=True)
        self.fc = layers.Dense(num_mixture)
        self.fc_z = layers.Dense(dec_rnn_size)

    def call(self, x, z):
        z_input = self.fc_z(z)
        x = tf.concat([x, tf.tile(tf.expand_dims(z_input, 1), [1, tf.shape(x)[1], 1])], axis=-1)
        out = self.rnn(x)
        out = self.fc(out)
        return out


# SketchRNN Model
class SketchRNN(Model):
    def __init__(self, num_classes=345, enc_rnn_size=256, dec_rnn_size=512, z_size=128, num_mixture=20):
        super(SketchRNN, self).__init__()
        self.encoder = Encoder(enc_rnn_size, z_size)
        self.decoder = Decoder(dec_rnn_size, num_mixture, z_size)
        self.fc = layers.Dense(num_classes)

    def call(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(x, z)
        out = self.fc(out)
        return out

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape)
        return eps * tf.exp(logvar * .5) + mu


class Solution:
    def __init__(self):
        self.num_classes = 345  # Assuming there are 345 classes, modify as needed
        self.model = SketchRNN(num_classes=self.num_classes)
        # Assuming the model weights are loaded somewhere, if needed

    # This is a signal that a new drawing is about to be sent
    def new_case(self):
        pass

    # Given a stroke, return a string of your guess
    def guess(self, x: list[int], y: list[int]) -> str:
        strokes = [x, y]
        image = strokes_to_image(strokes)
        # Convert the image to a tensor
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        outputs = self.model(image_tensor)
        predicted = tf.argmax(outputs, axis=1).numpy()
        # Return the label name based on the prediction
        # Here, you'd map the predicted index to its corresponding label name
        return self.label_to_name[predicted[0]]

    # This function is called when you get
    def add_score(self, score: int):
        print(score)
