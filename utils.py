#! /usr/bin/env python

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
from sacremoses import MosesTokenizer
tf.enable_eager_execution()


target_labels = ['comp', 'rec', 'sci', 'talk']


def get_datasets(max_sequence_length=10000, min_frequency=2):
    train = fetch_20newsgroups(subset='train', shuffle=True, random_state=57, remove=('header', 'footer', 'quates'))
    test = fetch_20newsgroups(subset='test', shuffle=True, random_state=57, remove=('header', 'footer', 'quates'))

    tokenizer = MosesTokenizer()
    train_target = []
    train_text = []
    for target, text in zip(train.target, train.data):
        for label in target_labels:
            if train.target_names[target].startswith(label):
                target = np.zeros(len(target_labels), dtype=np.float32)
                target[target_labels.index(label)] = 1.0
                train_target.append(target)
                train_text.append([token.lower() for token in tokenizer.tokenize(text)])
                break
    train_target = np.array(train_target, dtype=np.float32)

    test_target = []
    test_text = []
    for target, text in zip(test.target, test.data):
        for label in target_labels:
            if test.target_names[target].startswith(label):
                target = np.zeros(len(target_labels), dtype=np.float32)
                target[target_labels.index(label)] = 1.0
                test_target.append(target)
                test_text.append([token.lower() for token in tokenizer.tokenize(text)])
                break
    test_target = np.array(test_target, dtype=np.float32)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        max_sequence_length, min_frequency=min_frequency)
    corpus = train_text + test_text
    corpus = [' '.join(tokens) for tokens in corpus]
    vocab_processor.fit(corpus)

    train_text = np.array(list(vocab_processor.transform([' '.join(tokens) for tokens in train_text])), dtype=np.int32)
    test_text = np.array(list(vocab_processor.transform([' '.join(tokens) for tokens in test_text])), dtype=np.int32)

    train = tf.data.Dataset.from_tensor_slices({"target": train_target, "text": train_text})
    test = tf.data.Dataset.from_tensor_slices({"target": test_target, "text": test_text})

    return (train, test, len(target_labels), len(vocab_processor.vocabulary_))


if __name__ == '__main__':
    _, _, _, _ = get_datasets()
