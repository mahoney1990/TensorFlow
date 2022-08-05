# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 12:23:11 2022

@author: mahon
"""

import tensorflow as tf
import numpy as np
print("TF Version:", tf.__version__ )

#
mnist=tf.keras.datasets.mnist

#Xs are 28x28 arrays. We use the implied features of these arrays to predict
#A categorical assignment inherent in the Ys. 

#Here the X arrays reprsent pixel shanding on a 28x28 grid. Ys are letters
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Convert Xs to floats
x_train, x_test = x_train/255.0, x_test/255.0 

#Build predictive model. The NN will capture a bunch of 'hidden' relations between the
#position of X floats. First we define a NN with

model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)  
    ])


predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()
np.bincount(y_train)/60000

#Define loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

#Define optimization options 
model.compile(optimizer='SGD',
              loss=loss_fn,
              metrics=['accuracy'])

#Fit model
model.fit(x_train, y_train, epochs=5)

#Compare model predictions to test data set
model.evaluate(x_test, y_test, verbose=2)




























