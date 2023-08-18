import string
import numpy as np
from nltk import word_tokenize


class TextCounter():

    def __init__(self):
        self.count_types = [
            CountTypes.CHAR,
            CountTypes.PUNC,
            CountTypes.NUM,
            CountTypes.UPPER
        ]
        self.count_methods = [
            self._get_char_count,
            self._get_punc_count,
            self._get_numeric_count,
            self._get_uppercase_count
        ]

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        return self.count_features(X)

    def transform(self, X):
        return self.count_features(X)

    def count_features(self, X):
        # generates features dict to store values in lists organized by count type
        features = {}
        for count_type in self.count_types:
            features[count_type] = []

        # for each body of text, each count method is performed and values are stored in lists in the features dict organized by count type
        for text in X:
            for count_method in self.count_methods:
                count_type, count_val = count_method(text)
                features[count_type].append(count_val)

        # converts dict into 2d list
        features_as_2d_list = []
        for count_vals in features.values():
            features_as_2d_list.append(count_vals)

        # returns array with data organized by row instead of by column
        return np.array(features_as_2d_list).transpose()

    def _get_char_count(self, s: str) -> int:
        return (CountTypes.CHAR, len(s))

    def _get_punc_count(self, s: str) -> int:
        count = 0
        for char in s:
            count += 1 if char in string.punctuation else 0
        return (CountTypes.PUNC, count)

    def _get_numeric_count(self, s: str) -> int:
        count = 0
        tokens = word_tokenize(s)
        for word in tokens:
            count += 1 if word.isnumeric() else 0
        return (CountTypes.NUM, count)

    def _get_uppercase_count(self, s: str) -> int:
        count = 0
        for char in s:
            count += 1 if char.isupper() else 0
        return (CountTypes.UPPER, count)


class CountTypes:
    CHAR = 'char_count'
    PUNC = 'punc_count'
    NUM = 'numeric_count'
    UPPER = 'upper_count'
    TYPES = [CHAR, PUNC, NUM, UPPER]