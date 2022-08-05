# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 08:17:33 2022

@author: mahon
"""

import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf

import utils

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile

import re

AUTOTUNE=tf.data.AUTOTUNE
seed=1337
tf.get_logger().setLevel('INFO')
#%% Grab War and Peace, split out text by chapter, tokenize and clean.
with open(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\tolstoy.txt', encoding="utf-8") as f:
    lines = f.readlines()
    
res = [sub.split() for sub in lines]    


N=len(res)

lines=[]
chap_index=[]
chap_text=[]
text_list=[]

#Combine lines of text into 
for i in range(N):
    if res[i]==[]:
        continue
    else:
        lines.append(' '.join(res[i]))    

for i in range(len(lines)):
    lines[i] = lines[i].strip()
    lines[i]=lines[i].encode('utf-8',errors='replace').decode('utf-8','replace')
    lines[i] = re.sub(r'[^\w\s]', ' ', lines[i])
    lines[i] = re.sub('  ', ' ', lines[i])

#Find occurances of 'CHAPTER'
for i, j in enumerate(lines):
    if 'CHAPTER' in j :
        chap_index.append(i)

#Split by chapter index -- text is one string variable per chapter
for j in range(len(chap_index)-1):
    chap_text.append('')
    for i in range(chap_index[j],chap_index[j+1]):
        chap_text[j]=chap_text[j]+' '+lines[i]
        
#Text is list for each chapter. text_list is a list of lists  
for j in range(len(chap_index)-1):
        text_list.append(lines[chap_index[j]:chap_index[j+1]])
        
#Grab Chapter One, Two, Three, and Four
for i in range(len(text_list)):
    text += [text for text in text_list[0]]

#Turn list of strings into tf dataset object -- initate the black box!
text_ds = tf.data.Dataset.from_tensor_slices(text).filter(lambda x: tf.cast(tf.strings.length(x),bool))

#Build a standardization function to remove junk
#Lower case everything, take out puctuation (already did that, but hey lets do it again)
def custom_standardization(input_data):
    lowercase=tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,'[%s]' %  re.escape(string.punctuation), '')

#define vocab size and sequence length
vocab_size=2000
sequence_length=5

#Initialize  Vectorization layer -- this thing will conver our text data to vectors automatically!
vectorize_layer = layers.experimental.preprocessing.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

#Call the vectorize layer
vectorize_layer.adapt(text_ds.batch(1000))

#Get list of words from the vectorize layer
#inverse_vocab=vectorize_layer.get_vocabulary() <---Fuck this thing

def _get_vocabulary():
    keys, values = vectorize_layer._index_lookup_layer._table_handler.data()
    return [x.decode('utf-8', errors='ignore') for _, x in sorted(zip(values, keys))]

inverse_vocab=_get_vocabulary()
print(len(inverse_vocab))

#%%
#text
text_vector_ds=text_ds.batch(120).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

#Convert to list of sequences -- now we can run this thing like our simple example above!
sequences=list(text_vector_ds.as_numpy_iterator())
print(len(sequences))

#Lets check it out! **Steve Brule voice**
for seq in sequences[5:10]:
    print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

###Now, lets actually build our training dataset using these sequences
### Run that function! Should get three arrays -- one for our target indexes,
### one for context (first entry in each is actual context, the other four are negative samples),
### the label array that sez first entry in context is 1, others 0 

num_ns=10

targets, contexts, labels = generate_training_data(
    sequences, 
    window_size=10,
    num_ns=num_ns,
    vocab_size=vocab_size,
    seed=1337)

targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")
 
####Okay, lets actually estimate a NN model
#First lets specify the data input object

BATCH_SIZE=12800
BUFFER_SIZE=10000

dataset=tf.data.Dataset.from_tensor_slices(((targets, contexts),labels))
dataset=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=False)

dataset=dataset.cache().prefetch(buffer_size=AUTOTUNE)

#Now lets get to trainin'
#We'll define our own tf model as a class to make this thing airtight
#wtf is dots doing?

class word_to_vec(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim):
        super(word_to_vec, self).__init__()
        
        self.target_embedding = layers.Embedding(
            vocab_size,
            embedding_dim, 
            input_length=1, 
            name='w2v_embedding')

        self.context_embedding = layers.Embedding(
            vocab_size,
            embedding_dim, 
            input_length=num_ns+1)
    
    def call(self, pair):
        target, context = pair 
        
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis = 1)
            
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)

        dots =  tf.einsum('be,bce->bc', word_emb, context_emb)

        return dots


def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits = x_logit, labels=y_true)

loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)

embedding_dim=120
word2vec=word_to_vec(vocab_size, embedding_dim)

word2vec.compile(optimizer='adam',
                 loss=loss,
                 metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='/tmp/me/logs')

word2vec.fit(dataset,epochs=20,callbacks=[tensorboard_callback])

#Export to visualizer
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

import io

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  print(index, word)
  if index == 0:
   continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()

#goto https://projector.tensorflow.org/ and upload 'metadata.tsv' for PCA visualzation
try:
  from google.colab import files
  files.download('vectors.tsv')
  files.download('metadata.tsv')
except Exception:
  pass




