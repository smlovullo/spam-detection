import os
import logging
import joblib
from datetime import datetime

from sklearn.pipeline import Pipeline

from src.utils.constants import LoggerConstants, ModelConstants, IOConstants

class Classifier:

    def __init__(self) -> None:
        self._logger = logging.Logger(LoggerConstants.CLASSIFY_LOGGER_NAME, level=logging.INFO)
        self._logger.addHandler(logging.FileHandler(f'{LoggerConstants.LOGS_DIR}{LoggerConstants.CLASSIFY_LOG_FILE_BASE_NAME}-{datetime.now().strftime("%Y%m%d-%H%M%S%f")}.log'))
        self._logger.info('Initializing Classifier...')
        self.errors = []

        self._classifier = self._import_classifier(f"{ModelConstants.MODELS_DIR}{ModelConstants.MODEL_BASE_NAME}.joblib")

        self._logger.info('Classifier initialization complete')

    def _import_classifier(self, file_path: str) -> Pipeline:
        try:
            classifier = joblib.load(file_path)
            if isinstance(classifier, Pipeline):
                return classifier
            else:
                raise ClassifierNotFoundError
        except Exception as e:
            msg = repr(e)
            self._logger.error(msg)
            self.errors.append(msg)
            return None

    def classify(self, message: str) -> tuple:
        if self._classifier is None:
            errors = self.errors
            self.errors = []
            return (None, errors)
        
        try:
            self._logger.info(f'Begin classifying message: {message}')

            y_hat = self._classifier.predict([message]).tolist()
            y_hat = [ClassificationTypes.SPAM if prediction == 1 else ClassificationTypes.HAM for prediction in y_hat]

            y_hat_proba_raw = self._classifier.predict_proba([message]).tolist()
            y_hat_proba = []
            for i, prediction in enumerate(y_hat):
                if prediction == ClassificationTypes.SPAM:
                    y_hat_proba.append(y_hat_proba_raw[i][1])
                else:
                    y_hat_proba.append(y_hat_proba_raw[i][0])

            result = {'prediction': y_hat, 'probability': y_hat_proba}
        except Exception as e:
            msg = repr(e)
            self._logger.error(msg)
            self.errors.append(msg)
            result = None

        errors = self.errors
        self.errors = []
        return (result, errors)


class ClassificationTypes:
    SPAM = 'spam'
    HAM = 'ham'


class ClassifierNotFoundError(Exception):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        msg = "Unexpected data type found when reading model data from file, file is not formatted as readable model data."
        return msg