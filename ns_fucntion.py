# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:05:59 2022

@author: mahon
"""

import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

def generate_training_data(sequences, window_size,num_ns,vocab_size,seed):
    
    #
    targets, contexts, labels =[], [], []
    
    #Build Sampling Table
    sampling_table=tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    
    #iterate over sentences in dataset.
    for sequence in tqdm.tqdm(sequences):
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence, 
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        for target_word, context_word in positive_skip_grams:
    
            context_class=tf.expand_dims(
                tf.constant([context_word],dtype='int64'), 1)
    
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class, 
                num_true=1, 
                num_sampled=num_ns, 
                unique=True, 
                range_max=vocab_size,
                seed=seed,
                name='negative_sampling')

            negative_sampling_candidates=tf.expand_dims(negative_sampling_candidates,1)
        
            context=tf.concat([context_class,negative_sampling_candidates], 0)
            label=tf.constant([1]+[0]*num_ns, dtype='int64')
            
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
        
    return targets, contexts, labels 













