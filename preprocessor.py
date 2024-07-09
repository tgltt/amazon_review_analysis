import numpy as np
import nltk

from sklearn.decomposition import PCA

from kneed import KneeLocator

from vocab import Voc

from consts import MAX_CONTENT_LENGTH
from consts import UNK

class PreProcessor:

    def __init__(self):
        pass

    def filter_stopwords_(self, X):
        for x in X:
            for col in range(2):
                content = x[col]
                origin_text_words = nltk.word_tokenize(content.strip().lower())

                new_texts = []
                for word in origin_text_words:
                    if word not in self.voc_model.stopwords:
                        new_texts.append(word)

                x[col] = " ".join(new_texts)

        return X

    def deal_input(self, X):
        """
        将title和text进行拼接
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        sep_col = np.empty(shape=len(X)).astype(np.str_)
        sep_col[:] = " "

        X_merge = np.char.add(X[:, 0], sep_col)
        X_merge = np.char.add(X_merge, X[:, 1])

        return X_merge

    def construct_vocab(self, X):
        X = self.deal_input(X)

        vocab, word2idx, idx2word, stopwords = Voc.construct_vocab(X)
        self.voc_model = Voc(vocab, word2idx, idx2word, stopwords)

        return self.voc_model

    def vectorize(self, X, y=None):
        if X is None:
            raise Exception("X is illegal")

        return self._vectorize_data(X)

    def _vectorize_data(self, X):
        vocab = self.voc_model.vocab
        word2idx = self.voc_model.word2idx

        X_deal = self.deal_input(X)

        sentence_count, vocab_size = len(X_deal), len(vocab)

        X_vec = np.zeros((sentence_count, vocab_size), dtype=np.int32)
        for sentence_idx in range(sentence_count):
            x_deal = X_deal[sentence_idx][:MAX_CONTENT_LENGTH]
            words = nltk.word_tokenize(x_deal.strip().lower())
            for word in words:
                word_idx = word2idx[word] if word in vocab else word2idx[UNK]
                X_vec[sentence_idx][word_idx] += 1

        X_vec = np.concatenate((X_vec, X[:, 2:].astype(np.float32)), axis=-1)
        X_vec = X_vec.astype(np.int32)

        return X_vec

    def transform_data(self, samples):
        samples = self.filter_stopwords_(samples)

        content = self.vectorize(samples)
        content = self.pca.transform(content)

        return content

    def init_PCA(self, X):
        if "pca" in self.__dict__:
            return self.pca

        pca = PCA()
        pca.fit(X)

        kl = KneeLocator(range(1, len(pca.explained_variance_) + 1),
                         pca.explained_variance_,
                         S=10.0,
                         curve='convex',
                         direction='decreasing')

        self.pca = PCA(n_components=kl.elbow, svd_solver="auto")
        X_pca = self.pca.fit_transform(X)

        print(f"本次降维保留了{X_pca.shape[1]}个特征，累积方差解释比例是{self.pca.explained_variance_ratio_.sum():.4}")

        return self.pca