# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:28:55 2022

@author: mahon
"""

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing

#Supress debugging information 
tf.get_logger().setLevel('INFO')

print(tf.__version__)

#Extract movie review data set
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

#Define reference for full dataset
dataset_dir=os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)

#Define reference for training dataset
train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)

#Grab sample file
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
  print(f.read())

#Remove unnessesary shit
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

#Now that there are only two subdirectories -- pos and neg -- we sample them to produce
#our training data set
batch_size = 32
seed = 42

#Grab files, with
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)

#Look at some data
for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i])
    print("Label", label_batch.numpy()[i])

raw_train_ds.class_names[0]
raw_train_ds.class_names[1]

#Build raw test and validation sets
raw_val_ds=tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

raw_test_ds=tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    )


#Define a function to conver text to LC and replace HTML shit
def standardize_data(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped = tf.strings.regex_replace(lowercase, '<br />','')
    return stripped

#Need a text vectorization layer
max_features=10000
seq_length=250

vectorize_layer=layers.experimental.preprocessing.TextVectorization(
    standardize=standardize_data,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=seq_length    
    )

#Make a text-only dataset
train_text=raw_train_ds.map(lambda x,y:x)
vectorize_layer.adapt(train_text)

def vectorize_text(text,label):
    text=tf.expand_dims(text, -1)
    return vectorize_layer(text), label

text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = standardize_data(text_batch[1]), label_batch[1]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

vectorize_layer.get_vocabulary()[8999]

#Apply the TextVectorazation Layer to train, test sets
train_ds=raw_train_ds.map(vectorize_text)
val_ds=raw_val_ds.map(vectorize_text)
test_ds=raw_test_ds.map(vectorize_text)


###
AUTOTUNE=tf.data.AUTOTUNE

#Set cache and pre-fetch settings
train_ds=train_ds.cache().prefetch(AUTOTUNE)
val_ds=val_ds.cache().prefetch(AUTOTUNE)
test_ds=test_ds.cache().prefetch(AUTOTUNE)

#Lets build this motherfucker
embedding_dim=16

model = tf.keras.Sequential([
    layers.Embedding(max_features+1, embedding_dim),
    layers.Dropout(.20),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(.20),
    layers.Dense(1)
    ])

model.summary()

#Compile the model. I.e give it an optim routine and
#a loss function
model.compile(optimizer='adam',
              loss=losses.BinaryCrossentropy(from_logits=True),
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    
#Train the model
epochs=10

history=model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)






