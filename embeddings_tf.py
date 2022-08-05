# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:04:21 2022

@author: mahon
"""

#####Lets build some word embeddings!

import io
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras import preprocessing

tf.get_logger().setLevel('INFO')

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

batch_size=1024
seed=123

train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=.20,
    subset='training',
    seed=seed
    )

val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=.20,
    subset='validation',
    seed=seed
    )

for text_batch, label_batch  in train_ds.take(1):
    for i in range(5):
        print(label_batch[i].numpy(), text_batch.numpy()[i])

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Build embeddings
embedding_layer=tf.keras.layers.Embedding(1000, 5)











