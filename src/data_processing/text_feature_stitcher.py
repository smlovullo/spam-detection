import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler

from src.data_processing.text_cleaner import TextCleaner
from src.data_processing.text_counter import TextCounter, CountTypes


class TextFeatureStitcher():

    def __init__(self) -> None:
        self._text_counter_pipe = Pipeline([
            ('counter', TextCounter()),
            ('scaler', RobustScaler())
        ])
        self._tfidf_bow_pipe = Pipeline([
            ("cleaner", TextCleaner()),
            ("vectorizer", TfidfVectorizer())
        ])

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        counts_df = pd.DataFrame(self._text_counter_pipe.fit_transform(X), columns=CountTypes.TYPES)

        tdidf_bow = self._tfidf_bow_pipe.fit_transform(X).toarray()
        vocab = self._tfidf_bow_pipe.named_steps["vectorizer"].get_feature_names_out()
        words_df = pd.DataFrame(tdidf_bow, columns=vocab)

        return self._combine_results(counts_df, words_df)

    def transform(self, X):
        counts_df = pd.DataFrame(self._text_counter_pipe.transform(X), columns=CountTypes.TYPES)

        tdidf_bow = self._tfidf_bow_pipe.transform(X).toarray()
        vocab = self._tfidf_bow_pipe.named_steps["vectorizer"].get_feature_names_out()
        words_df = pd.DataFrame(tdidf_bow, columns=vocab)

        return self._combine_results(counts_df, words_df)

    def _combine_results(self, counts_table: pd.DataFrame, words_table: pd.DataFrame) -> np.ndarray:
        joined_df = counts_table.join(words_table)
        return joined_df.values