# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 08:15:03 2022

@author: mahon
"""

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

#%%
#Practice One

abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_train.head()

#Goal is to predict age from other features
abalone_features=abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

abalone_features=np.array(abalone_features)
abalone_features

#Simple NN
abalone_model=tf.keras.Sequential([
    layers.Dense(16),
    layers.Dropout(.20),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(.20),
    layers.Dense(1)])

abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

abalone_model.fit(abalone_labels,abalone_features,epochs=10)









#%%
#Practice Two