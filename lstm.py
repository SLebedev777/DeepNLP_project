# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:43:04 2018

"""
from __future__ import print_function

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
from sklearn.utils import class_weight, shuffle
from sklearn.preprocessing import normalize

os.chdir('CASES_DATA')
df_train = pd.read_csv('cases_train_morph_norm.txt', sep='\t', encoding='cp1251')
df_test = pd.read_csv('cases_test_morph_norm.txt', sep='\t', encoding='cp1251')
print(len(df_train))
print(len(df_test))


def balance_dataset(df, n):
    print('before balancing: ')
    print(df['label'].value_counts())
    class1 = df[df['label'] == 1]
    print(len(class1))
    for i in range(n):
        df = df.append(class1)
    print('after balancing: ')
    print(df['label'].value_counts())
    return shuffle(df)

print('balancing train')
df_train = balance_dataset(df_train, 6)
print('balancing test')
df_test = balance_dataset(df_test, 7)


print('splitting data')
data_train, y_train = df_train['text'].values, df_train['label'].values
data_test, y_test = df_test['text'].values, df_test['label'].values



print('tokenize text')
MAXLEN = 3000   # length of sequence
NUMWORDS = 10000
t = Tokenizer(num_words=NUMWORDS)
t.fit_on_texts(data_train)
vocab_size = len(t.word_index) + 1
if vocab_size >= NUMWORDS:
    vocab_size = NUMWORDS
# integer encode the documents
X_train = t.texts_to_sequences(data_train)
X_train = pad_sequences(X_train, maxlen=MAXLEN)
X_test = t.texts_to_sequences(data_test)
X_test = pad_sequences(X_test, maxlen=MAXLEN)


use_pretrained_wordvectors = True

if use_pretrained_wordvectors:
    from gensim.models import KeyedVectors
    wv = KeyedVectors.load('w2v_morph_norm.txt')
    embedding_size = wv.vector_size
    print('loaded pretrained gensim word vectors:')
    print('vocab_size = %d  embedding_size= %d' %(len(wv.vocab), embedding_size))
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word, i in t.word_index.items():
        if i >= vocab_size:
            break
        if word not in wv:
            continue
        embedding_vector = wv[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    #embedding_matrix = normalize(embedding_matrix)
else:
    embedding_size = 32

print('Building model...')
model = Sequential()
if use_pretrained_wordvectors:
    model.add(Embedding(vocab_size, embedding_size,
                            weights=[embedding_matrix],
                            input_length=MAXLEN,
                            mask_zero=False,
                            trainable=False))
else:
    model.add(Embedding(vocab_size, embedding_size, input_length=MAXLEN, mask_zero=False))
model.add(Dropout(0.2))
model.add(Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''
class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                  classes=[0, 1], y=y_train)
class_weights = dict(enumerate(class_weights))
sample_weights = class_weight.compute_sample_weight(class_weights, y_train)
'''

batch_size = 16
epochs = 7

# callbacls
early_stopping = EarlyStopping('val_loss')
model_checkpoint = ModelCheckpoint(filepath='model_{epoch:02d}_{val_acc:.2f}',
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True)
                                  
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    class_weight=None,
                    sample_weight=None,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping, model_checkpoint])
score = model.evaluate(X_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('predicting')
y_pred_test = binarize(model.predict(X_test), 0.5)

print('On test:')
print(metrics.classification_report(y_test, y_pred_test))


