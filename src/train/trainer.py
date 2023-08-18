import os
import re
import logging
from typing import List
from datetime import datetime
import yaml
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from src.data_processing.text_feature_stitcher import TextFeatureStitcher
from src.utils.constants import ConfigConstants, DataConstants, LoggerConstants, ModelConstants, IOConstants


class Trainer:
    """Trainer for spam detection classification model.

    A Trainer is responsible for carrying out the training of a spam classification model.
    It's main method that carries out this task is `train`.

    Although Trainer has no parameters for initialization, it does require that the config
    and raw message data files are in fixed locations specified by `CONFIG_PATH` and `DATA_PATH`
    in `utils/constants.py`. These files are read upon initialization.
    """

    def __init__(self) -> None:
        self._logger = logging.Logger(LoggerConstants.TRAIN_LOGGER_NAME, level=logging.DEBUG)
        self._logger.addHandler(logging.FileHandler(f'{LoggerConstants.LOGS_DIR}{LoggerConstants.TRAIN_LOG_FILE_BASE_NAME}-{datetime.now().strftime("%Y%m%d-%H%M%S%f")}.log'))
        self._logger.info('Initializing Trainer...')
        self.errors = []

        self._config = self._import_config(ConfigConstants.CONFIG_PATH)
        self._messages_df = self._import_text_data(DataConstants.DATA_PATH)

    def _import_config(self, file_path: str) -> dict:
        try:
            with open(file_path, IOConstants.READ) as file_stream:
                config = yaml.load(file_stream, yaml.SafeLoader)
        except Exception as e:
            msg = repr(e)
            self._logger.error(msg)
            self.errors.append(msg)
            config = None

        self._logger.info('Trainer initialization complete.')
        return config

    def _import_text_data(self, file_path: str) -> pd.DataFrame:
        try:
            with open(file_path, IOConstants.READ) as file_stream:
                sms_messages_raw = file_stream.readlines()
            labels = [re.search("(^.*)\t", message).group(1) for message in sms_messages_raw]
            sms_messages = [message[message.index('\t')+1:-1] for message in sms_messages_raw]
            messages_df = pd.DataFrame(data={DataConstants.DATA_LABELS_HEADER: labels, DataConstants.DATA_TEXT_HEADER: sms_messages})
            messages_df[DataConstants.DATA_LABELS_HEADER] = messages_df[DataConstants.DATA_LABELS_HEADER].apply(lambda x: 1 if x == 'spam' else 0)

        except Exception as e:
            msg = repr(e)
            self._logger.error(msg)
            self.errors.append(msg)
            messages_df = None

        return messages_df

    def train(self) -> List[str]:
        """Primary method to train a new model and save it to a file.

        This method will run the entire training process. It will return a list of errors
        that occured while trying to train a new model. If no errors occured, then an
        empty list will be returned and a model file will be written in the directory
        `/models/`. If there were errors, then a new model will not be created.

        The saved model file is a Scikit-Learn Pipeline object that utilizes the custom
        class `TextCounter`. The created models files are not designed for use outside of
        this application.
        """
        if self._config is None or self._messages_df is None:
            return self.errors

        try:
            self._logger.info('Beginning model training...')
            classifier = Pipeline([
                ('feature_stitcher', TextFeatureStitcher()),
                ('classifier', MLPClassifier(
                    hidden_layer_sizes=self._config[ConfigConstants.CONFIG_HP][ConfigConstants.CONFIG_HLS],
                    activation=self._config[ConfigConstants.CONFIG_HP][ConfigConstants.CONFIG_ACTIVATION],
                    solver=self._config[ConfigConstants.CONFIG_HP][ConfigConstants.CONFIG_SOLVER],
                    alpha=self._config[ConfigConstants.CONFIG_HP][ConfigConstants.CONFIG_ALPHA],
                    learning_rate=self._config[ConfigConstants.CONFIG_HP][ConfigConstants.CONFIG_LR],
                    max_iter=self._config[ConfigConstants.CONFIG_HP][ConfigConstants.CONFIG_MAX_ITER]
                ))
            ])

            classifier.fit(self._messages_df[DataConstants.DATA_TEXT_HEADER], self._messages_df[DataConstants.DATA_LABELS_HEADER])
            self._logger.info('Model is trained. Writing to file...')

            file_path = f"{ModelConstants.MODELS_DIR}{ModelConstants.MODEL_BASE_NAME}.joblib"
            joblib.dump(classifier, file_path)
            self._logger.info(f'Trained classification model saved at: {os.path.abspath(file_path)}')

        except Exception as e:
            msg = repr(e)
            self._logger.error(msg)
            self.errors.append(msg)

        return self.errors