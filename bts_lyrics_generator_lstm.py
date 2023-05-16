# -*- coding: utf-8 -*-
"""bts-lyrics-generator-starter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LMec7KhefNbgbI3ZvPSfQi_DJxg5ksQI
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import string

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

"""# Data Loading"""


#set the data paths ( ++ adjust for your system ++)

df = pd.read_csv("bts-noutf.csv")


#convert to lowercase"""

df['lyrics'] = df['lyrics'].apply(lambda x: str(x).lower())

bts_lyrics = []
for lyrics in df['lyrics']:
    bts_lyrics.append(lyrics)

new_lyrics = []
bts_lyrics = bts_lyrics[:-1]

for lyric in bts_lyrics:
    for line in lyric.split('\n'):
        new_lyrics.append(line)

# Remove Punctuations from the lyrics"""

final_lyrics = []
for lyrics in new_lyrics:
    final_lyrics.append(lyrics.translate(str.maketrans('', '', string.punctuation)))

final_lyrics = list(set(final_lyrics))

final_lyrics = final_lyrics[1:]

# Tokenize Lyrics"""

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(final_lyrics)

word_index = tokenizer.word_index
print(len(word_index))

sqs = tokenizer.texts_to_sequences(final_lyrics[1:])
padded = pad_sequences(sqs, padding='pre')
print(final_lyrics[0])
print(padded[0])

padded.shape

max_len = max([len(x) for x in sqs])


xs, labels = padded[:, :-1], padded[:, -1]
total_words = len(tokenizer.word_index) + 1
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Create a Bi-directional LSTM

model = Sequential()
model.add(Embedding(total_words, 128, input_length=max_len-1))
model.add(Bidirectional(LSTM(200, return_sequences=True)))
model.add(Bidirectional(LSTM(75)))
model.add(Dense(total_words, activation='relu'))
adam = Adam(learning_rate=0.002)

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
cb = ModelCheckpoint('best.h5', save_best_only=True)
lrr= ReduceLROnPlateau(monitor='accuracy', factor=.1, patience=5, min_lr=1e-5)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1, callbacks=[es, cb, lrr])

import pickle as pl
pl.dump(model,open('model.pkl','wb'))



