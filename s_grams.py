# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:13:19 2022

@author: mahon
"""

import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

SEED=42
AUTOTUNE=tf.data.AUTOTUNE


sentence = "The wide road shimmered in the hot sun"

tokens=list(sentence.lower().split())
print(len(tokens))

vocab, index = {}, 1 #start index at 1
vocab['<pad>'] = 0

for token in tokens:
    if token not in vocab:
        vocab[token] = index
        index+=1

vocab_size=len(vocab)

#invert the vocab dictionary
inverse_vocab={index: token for token, index in vocab.items()}

#Vectorize the sentence:
example_sentence=[vocab[word] for word in tokens]

window_size=1

#Positive sampling, construct word pairs that are next to each other
#We pick one target and one contact words, then sample other words for our estimator
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
    example_sentence, 
    vocab_size,
    window_size=window_size,
    negative_samples=0)

#Target word is the used word, context word is the one we want associated with the target
target_word, context_word = positive_skip_grams[0]
print(inverse_vocab[target_word])
print(inverse_vocab[context_word])

#4 is a good number of NS draws for a large vocab
num_ns = 4

#Build a singleton tensor containing the context word index (i.e. an integer)
context_class = tf.reshape(tf.constant(context_word, dtype='int64'), (1,1))

#Construct negative sampler
negative_sampling_candidates, _, _=tf.random.log_uniform_candidate_sampler(
    true_classes=context_class, 
    num_true =1 , 
    num_sampled=num_ns, 
    unique=True, 
    range_max=vocab_size,
    seed=SEED,
    name='Negative_Sampling')

print(negative_sampling_candidates)
print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])

# Add a dimension so you can use concatenation (in the next step).
negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

#Concatenate a positive context word with negative sampled words.
context=tf.concat([context_class, negative_sampling_candidates], axis=0)
print(context)

#Label the first context word as '1' and the rest as '0', this is our outcome vector
#The word attached to '1' is relvent, the others are not. Embeddings should try to match this association
label=tf.constant([1]+[0]*num_ns, dtype='int64')

# Reshape the target to shape '(1,)' and context and label to '(num_ns,)
target=tf.squeeze(target_word)
context=tf.squeeze(context)
label=tf.squeeze(label)

#Preprocessing finished! Okay lets scale this thing up and build a model
sampling_table=tf.keras.preprocessing.sequence.make_sampling_table(size=10)

print(f"target_index    : {target}")
print(f"target_word     : {inverse_vocab[target_word]}")
print(f"context_indices : {context}")
print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
print(f"label           : {label}")

#%% Now lets try to scale this up to a bigger dataset and actually build a model

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

with open(path_to_file) as f:
  lines = f.read().splitlines()

#New text is the several Shakespere plays, remove blank lines
text_ds=tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x),bool))

#Build a standardization function to remove junk
#Lower case everything, take out puctuation
def custom_standardization(input_data):
    lowercase=tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,'[%s]' %  re.escape(string.punctuation), '')

#define vocab size and sequence length
vocab_size=4096
sequence_length=10

#Initialize  Vectorization layer -- this thing will conver our text data to vectors automatically!
vectorize_layer = layers.experimental.preprocessing.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

#Call the vectorize layer
vectorize_layer.adapt(text_ds.batch(1024))

#Get list of words from the vectorize layer
inverse_vocab=vectorize_layer.get_vocabulary()

#text
text_vector_ds=text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

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

targets, contexts, labels = generate_training_data(
    sequences, 
    window_size=2,
    num_ns=4,
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

BATCH_SIZE=1024
BUFFER_SIZE=10000

dataset=tf.data.Dataset.from_tensor_slices(((targets, contexts),labels))
dataset=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

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

embedding_dim=128
word2vec=word_to_vec(vocab_size, embedding_dim)

word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')

word2vec.fit(dataset,epochs=20,callbacks=[tensorboard_callback])

#Export to visualizer
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()


try:
  from google.colab import files
  files.download('vectors.tsv')
  files.download('metadata.tsv')
except Exception:
  pass


