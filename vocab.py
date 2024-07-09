import os
import numpy as np

import joblib
import nltk

from consts import UNK

# 构建词典并向量化文本(机器学习)
class Voc:
    def __init__(self, vocab, word2idx, idx2word, stopwords):
        self.vocab = vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.stopwords = stopwords

    def construct_vocab(X):
        vocab = {UNK}
        word2idx = {}
        idx2word = {}

#         stopwords = [line.strip() for line in open("./stopwords_eng.txt", "r", encoding="utf-8").readlines()]
        stopwords = []

        if isinstance(X, np.ndarray):
            X = X.tolist()

        for i in range(len(X)):
            x_word_set = set(nltk.word_tokenize(X[i].strip().lower()))
            count = 0
            for word in x_word_set:
                filter_word_set = set()
                if word in stopwords:
                    filter_word_set.add(word)
                    continue
                vocab = vocab.union(x_word_set - filter_word_set)

        print(f"Vocabulary has {len(vocab)} words")

        count = 0
        for word in vocab:
            word2idx[word] = count
            idx2word[count] = word
            count += 1

        return vocab, word2idx, idx2word, stopwords