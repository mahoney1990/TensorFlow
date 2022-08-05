# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 08:35:53 2022

@author: mahon
"""

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

#%%

data=pd.read_csv(r'C:\Users\mahon\Replication\Word_Count_Example.csv')

features=data.copy()
labels=features.pop('Party')
labels=np.array(labels)

features=np.array(features)
indexed_features=features[ : , 0:3]
features=np.delete(features, [0,1,2],axis=1)

features=features[:,1:100]
features = np.asarray(features).astype('float32')

#Build Model
pol_model=tf.keras.Sequential([
    layers.Dense(16),
    layers.Dropout(.20),
    layers.Dense(1),
    layers.Activation('sigmoid')]
    )

pol_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),optimizer='adam')

pol_model.fit(features, labels, epochs=20)


pol_model.predict(features)


#%%













