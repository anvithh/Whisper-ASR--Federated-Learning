import flwr as fl
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def getData(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]] < dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1
        
    return np.array(dx), np.array(dy)


model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

(x_train, y_train) , (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
dist = [0, 10, 10, 10, 4000, 3000, 4000, 5000, 10, 4500]
x_train, y_train = getData(dist, x_train, y_train)


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()
    
    def fit(self, parameters, client):
        model.set_weights(parameters)
        m = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        hist = m.history
        print("\nFit History : \n", hist, "\n")
        return model.get_weights(), len(x_train), {}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("\nGLOBAL Model Evaluation accuracy : ", accuracy, "\n")
        return loss, len(x_test), {"accuracy": accuracy}
    
fl.client.start_numpy_client(
    server_address="127.0.0.1:5010",

    client = FlowerClient()
)

