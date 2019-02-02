#! /usr/bin/env python

import numpy as np
import tensorflow as tf
from sacremoses import MosesTokenizer
tf.enable_eager_execution()


def get_datasets(max_sequence_length=1024, min_frequency=2):
    train = tf.data.experimental.CsvDataset('train.csv', [tf.string, tf.string], header=True)
    test = tf.data.experimental.CsvDataset('test.csv', [tf.string, tf.string], header=True)

    artist_participants = train.concatenate(test).map(lambda artist, text: artist).apply(tf.data.experimental.unique())
    artist_participants = [d.numpy().decode('utf-8') for d in artist_participants]

    tokenizer = MosesTokenizer()
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_sequence_length, min_frequency=min_frequency)
    corpus = train.concatenate(test).map(lambda artist, text: text)
    corpus = [' '.join(tokenizer.tokenize(d.numpy().decode('utf-8'))) for d in corpus]
    vocab_processor.fit(corpus)

    def make_ohe(artist):
        ohe = np.zeros(len(artist_participants), dtype=np.float32)
        ohe[artist_participants.index(artist.decode('utf-8'))] = 1.0
        return ohe

    def tokenize(text):
        tokens = tokenizer.tokenize(text.decode('utf-8'))
        tokens = list(vocab_processor.transform([' '.join(tokens)]))
        return tokens[0].astype(np.int32)

    def preprocess(artist, text):
        ohe = tf.py_func(make_ohe, [artist], [tf.float32])
        tokens = tf.py_func(tokenize, [text], [tf.int32])
        return (ohe, tokens)

    return (train.map(preprocess), test.map(preprocess), len(artist_participants), len(vocab_processor.vocabulary_))


if __name__ == '__main__':
    _, _, _, _ = get_datasets()
