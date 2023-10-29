# IMPORTANT
# unless you're willing to change the run.py script, keep the new_case, guess, and add_score methods.

import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers, Model

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
    def __init__(self, data_path="C:\\Users\\bunin\\Documents\\TAMUDatathon23\\quick_draw_data"):
        self.num_classes = len(os.listdir(data_path))
        self.label_to_name = {i: name.split('.')[0] for i, name in enumerate(os.listdir(data_path))}
        self.model = SketchRNN(num_classes=self.num_classes)
        # TODO: Load the model weights here if needed.

    # This is a signal that a new drawing is about to be sent
    def new_case(self):
        pass

    # Given a stroke, return a string of your guess
    def guess(self, x: list[int], y: list[int]) -> str:
        strokes = np.array([x, y]).astype(np.float32)
        # Convert the strokes to a tensor
        strokes_tensor = tf.convert_to_tensor(strokes, dtype=tf.float32)
        strokes_tensor = tf.expand_dims(strokes_tensor, axis=0)
        outputs = self.model(strokes_tensor)
        predicted = tf.argmax(outputs, axis=1).numpy()
        return self.label_to_name[predicted[0]]

    def add_score(self, score: int):
        print(score)

# You can initialize the solution like this:
solution = Solution()
