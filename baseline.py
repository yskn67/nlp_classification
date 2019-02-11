#! /usr/bin/env python

import numpy as np
import pandas as pd
import lightgbm as lgb
from sacremoses import MosesTokenizer
from gensim.models import FastText
from sklearn.model_selection import train_test_split


embedding_size = 32


def get_sentence_vector(text, size, tokenizer, ft_model):
    vec = np.zeros(size, dtype=np.float32)
    count = 0
    for word in tokenizer.tokenize(text):
        if word in ft_model.wv:
            norm = np.sum(ft_model.wv[word])
            vec += ft_model.wv[word] / norm
            count += 1
    if count != 0:
        vec /= count
    return vec


tokenizer = MosesTokenizer()

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

corpus = [tokenizer.tokenize(text) for df in (train, test) for text in df['text']]
ft_model = FastText(size=embedding_size, window=5, min_count=2, seed=57)
ft_model.build_vocab(sentences=corpus)
ft_model.train(sentences=corpus, total_examples=len(corpus), epochs=10)

labels = list({label for df in (train, test) for label in df['artist']})

X_train = np.array([get_sentence_vector(text, embedding_size, tokenizer, ft_model) for text in train['text']])
y_train = np.array([labels.index(label) for label in train['artist']])
X_test = np.array([get_sentence_vector(text, embedding_size, tokenizer, ft_model) for text in test['text']])
y_test = np.array([labels.index(label) for label in test['artist']])

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=57)

dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid)

params = {'objective': 'multiclass', 'num_class': len(labels), 'learning_rate': 0.0001, 'seed': 57}

bst = lgb.train(params, dtrain, num_boost_round=10000, valid_sets=[dtrain, dvalid], early_stopping_rounds=50)

corrects = []
for label, pred in zip(y_test, bst.predict(X_test)):
    if label == np.argmax(pred):
        corrects.append(1)
    else:
        corrects.append(0)
print('Accuracy:', sum(corrects) / len(corrects))
