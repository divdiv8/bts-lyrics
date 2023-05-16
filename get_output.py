import pickle as pl
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os


model_bts = pl.load(open('model_lstm.pkl', 'rb'))
#max_len = max([len(x) for x in sqs])
with open('btslyrics.txt', 'r', encoding='utf-8') as f:
    text = f.read()
bts_lyrics = text
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(bts_lyrics)
seed_text = "Run BTS"
next_words = 50

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], padding='pre')
    predicted = np.argmax(model_bts.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)