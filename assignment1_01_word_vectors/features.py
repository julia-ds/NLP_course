from collections import OrderedDict, defaultdict
from sklearn.base import TransformerMixin
from typing import List, Union
import numpy as np


class BoW(TransformerMixin):
    """
    Bag of words tranformer class
    
    check out:
    https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
    to know about TransformerMixin class
    """

    def __init__(self, k: int):
        """
        :param k: number of most frequent tokens to use
        """
        self.k = k
        # list of k most frequent tokens
        self.bow = None

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        # task: find up to self.k most frequent tokens in texts_train,
        # sort them by number of occurences (highest first)
        # store most frequent tokens in self.bow
        text = []
        for comment in X:
            text.extend(comment.lower().split(' '))
        vocab = set(text)
        bag_vec = np.zeros(len(vocab)) # created vocab of unique tokens and empty array for its frequencies
        
        words_index = OrderedDict() # saved index of a token, so I can come back to words later
        i = 0
        for token in vocab:
            words_index[token] = i 
            i += 1 
        
        for token in vocab:
            bag_vec[words_index[token]] = len([i for i in text if i == token])
            
        for i, key in enumerate(words_index.keys()):
            words_index[key] = bag_vec[i]
        self.bow = [k for k, v in sorted(words_index.items(), key=lambda item: item[1], reverse=True)[:self.k]]

        # fit method must always return self
        return self

    def _text_to_bow(self, text: str) -> np.ndarray:
        """
        convert text string to an array of token counts. Use self.bow.
        :param text: text to be transformed
        :return bow_feature: feature vector, made by bag of words
        """

        counts = defaultdict()
        result = np.zeros(len(self.bow))
        for token in text.lower().split(' '):
            if token in counts.keys():
                counts[token] += 1
            else:
                counts[token] = 1
        for i, token in enumerate(self.bow):
            if token in text.lower().split(' '):
                result[i] = counts[token]
            
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.bow is not None
        return np.stack([self._text_to_bow(text) for text in X])

    def get_vocabulary(self) -> Union[List[str], None]:
        return self.bow


class TfIdf(TransformerMixin):
    """
    Tf-Idf tranformer class
    if you have troubles implementing Tf-Idf, check out:
    https://streamsql.io/blog/tf-idf-from-scratch
    """

    def __init__(self, k: int = None, normalize: bool = False):
        """
        :param k: number of most frequent tokens to use
        if set k equals None, than all words in train must be considered
        :param normalize: if True, you must normalize each data sample
        after computing tf-idf features
        """
        self.k = k
        self.normalize = normalize

        # self.idf[term] = log(total # of documents / # of documents with term in it)
        self.idf = OrderedDict()

    def fit(self, X: np.ndarray, y=None):
        """
        :param X: array of texts to be trained on
        """
        N = len(X)
        text = []
        for comment in X:
            text.extend(comment.lower().split(' '))
        tf_idf_dict = OrderedDict() 
        for token in set(text):
            count_docs = 0
            tf = 0
            for comment in X:
                term_count = len([word for word in comment.lower().split(' ') if word == token]) 
                if term_count > 0:
                    count_docs += 1
                    tf += term_count / len(comment.lower().split(' '))
                
            tf_idf_dict[token]= tf * np.log(N/(count_docs + 1)) 
        
        if self.k:
            self.idf = {k: v for k, v in sorted(tf_idf_dict.items(), key=lambda item: item[1], reverse=True)[:self.k]}
        else:
            self.idf = {k: v for k, v in sorted(tf_idf_dict.items(), key=lambda item: item[1], reverse=True)}
        
        # fit method must always return self
        return self

    def _text_to_tf_idf(self, text: str) -> np.ndarray:
        """
        convert text string to an array tf-idfs.
        *Note* don't forget to normalize, when self.normalize == True
        :param text: text to be transformed
        :return tf_idf: tf-idf features
        """           
        result = np.zeros(len(self.idf))
        counts = OrderedDict()
        for token in text.lower().split(' '):
            if token not in counts.keys():
                if token in self.idf.keys():
                    counts[token] = self.idf[token]
        if self.normalize:
            norm_coeff = np.sqrt((np.array([e for e in counts.values()]) ** 2).sum())
            counts = {k: v/coef for k, v in counts.items()}
        else:
            counts = {k: v for k, v in counts.items()}
        for i, token in enumerate(self.idf):
            if token in counts.keys():
                result[i] = counts[token]
            
        return np.array(result, "float32")

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        :param X: array of texts to transform
        :return: array of transformed texts
        """
        assert self.idf is not None
        return np.stack([self._text_to_tf_idf(text) for text in X])
