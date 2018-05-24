# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:44:33 2018

"""
import os
import gensim, logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

os.chdir('CASES_DATA')
df = pd.read_csv('cases_train_morph_norm.txt', sep='\t', encoding='cp1251')
df = df.dropna()
print(len(df))

print('preparing data')
data_all, y_all = df['text'].values, df['label'].values
docs = [d.split() for d in data_all]

print('traning')
embedding_size = 256
# build vocabulary and train model
model = gensim.models.Word2Vec(
    docs,
    size=embedding_size,
    window=5,
    min_count=3,
    max_vocab_size=50000,
    workers=10,
    sg=1,
    compute_loss=True)
model.train(docs, total_examples=len(docs), epochs=15)
model.wv.save('w2v_morph_norm.txt')

embedding_matrix = np.zeros((len(model.wv.vocab), embedding_size))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

print('making 2d projection')
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

num_points = 1024
print('normalizing')
normalized_embeddings = normalize(embedding_matrix)


print('projecting')
reductor = TSNE(init='pca')
two_d_embeddings = reductor.fit_transform(normalized_embeddings[:num_points])

plt.figure(figsize=(30, 30))
words = [model.wv.index2word[i] for i in range(num_points)]

for i, label in enumerate(words):
    x, y = two_d_embeddings[i, :]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords="offset points",
                   ha="right", va="bottom")
plt.show()
