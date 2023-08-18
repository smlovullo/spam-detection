import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextCleaner():

    def __init__(self):
        self.punct_table = str.maketrans('', '', string.punctuation)
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.transformations = [
            self._strip_punct,
            self._convert_to_lowercase,
            self._remove_stopwords,
            self._remove_numbers,
            self._remove_special_characters,
            self._lemmatize
        ]

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        return self.clean_text(X)

    def transform(self, X):
        return self.clean_text(X)

    def clean_text(self, X):
        cleaned_text = []
        for text in X:
            for transformation in self.transformations:
                text = transformation(text)
            cleaned_text.append(text)
        return cleaned_text

    def _strip_punct(self, text: str) -> str:
        return text.translate(self.punct_table)

    def _convert_to_lowercase(self, text: str) -> str:
        return text.lower()

    def _remove_stopwords(self, text: str) -> str:
        words = word_tokenize(text)
        words = [w for w in words if w not in self.stopwords]
        return ' '.join(words)

    def _remove_numbers(self, text: str) -> str:
        words = word_tokenize(text)
        words = [w for w in words if not re.search(r'\d', w)]
        return ' '.join(words)

    def _remove_special_characters(self, text: str) -> str:
        pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)

    def _lemmatize(self, text: str) -> str:
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(w) for w in words]
        return ' '.join(words)