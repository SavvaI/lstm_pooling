#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import keras
import re
from keras.preprocessing import sequence
from keras.preprocessing import text
help(keras.layers.UpSampling1D)
class BatchGenerator:
    def __init__(self, datapath, batch_size, num_words, seq_len):
        self.batch_size = batch_size
        self.text = open(datapath, 'r', encoding='utf8').read()
        self.messages = [i for i in self.text.split('\n\n\n') if not '\t' in i and not 'Attachments:' in i]
        self.participants = [i.split(' (')[0].strip() for i in self.messages]
        self.participants = {i: j for i, j in enumerate(list(set(self.participants)))}
        # self.messages = [[j.strip() for j in re.sub(r'([^a-zA-Zа-яА-Я]+)', r' \1 ', i).split(' ')] for i in self.messages]
        self.data = [[' '.join(i.split('\n')[1:]).strip(), i.split(' (')[0].strip()] for i in self.messages]
        # self.messages = [' '.join(self.messages[i].split('\n')[1:]).strip(' \t\n') for i in range(len(self.messages))]
        self.x, self.y_ = zip(*self.data)
        self.tokenizer = text.Tokenizer(num_words=num_words, char_level=True, filters='')
        self.tokenizer.fit_on_texts(self.x)
        self.name2ind = {value: key for key, value in self.participants.items()}
        self.ind2name = self.participants
        # help(self.tokenizer.sequences_to_matrix)
        self.x = self.tokenizer.texts_to_sequences(self.x)
        def split_list_of_lists(l):
            return [[[j] for j in i] for i in l]
        self.x = [self.tokenizer.sequences_to_matrix(i) for i in split_list_of_lists(self.x)]
        # print(len(self.x), self.x[0].shape)
        self.x = sequence.pad_sequences(self.x, seq_len)
        print(self.x.shape, self.x[0, -1])
        self.y_ = [self.name2ind[i] for i in self.y_]
        self.y_ = np.stack(self.y_)

batch_size = 20
seq_len = 100
num_words = 50
stride = 5
layer_size = 100
# embed_dim = 100

def add_LSTM_pooling(m, size = 50, input_shape=None):
    if input_shape is not None:
        m.add(keras.layers.core.Reshape([-1, stride, input_shape[-1]], input_shape=input_shape))
    else:
        m.add(keras.layers.core.Reshape([-1, stride, m.output_shape[-1]]))
    m.add(keras.layers.TimeDistributed(keras.layers.LSTM(size)))



a = BatchGenerator('data/sushnost.txt', batch_size=batch_size, num_words=num_words, seq_len=seq_len)

m = keras.models.Sequential()
add_LSTM_pooling(m, layer_size, dinput_shape=[seq_len, num_words])
# m.add(keras.layers.Embedding(num_words, embed_dim))
# m.add(keras.layers.LSTM(50, input_shape=[seq_len, num_words], return_sequences=True))
# m.add(keras.layers.Lambda(lambda x: x[:,0: :5,:]))
# m.add(keras.layers.MaxPool1D(pool_size=5))
add_LSTM_pooling(m, layer_size)
m.add(keras.layers.LSTM(layer_size))
# m.add(keras.layers.MaxPool1D(pool_size=5))
# m.add(keras.layers.Lambda(lambda x: x[:,0: :5,:]))
m.add(keras.layers.Dense(num_words, activation=keras.activations.softmax))

m.compile(keras.optimizers.Adam(), keras.losses.sparse_categorical_crossentropy, [keras.metrics.sparse_categorical_accuracy])
m.fit(a.x, a.y_, batch_size=batch_size, epochs=3, shuffle=True)
