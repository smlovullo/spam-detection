class DataConstants:
    DATA_PATH = './data/SMSSpamCollection.txt'
    DATA_LABELS_HEADER = 'Labels'
    DATA_TEXT_HEADER = 'Messages'


class ConfigConstants:
    CONFIG_PATH = './config/train_config.yaml'
    CONFIG_HP = 'hyperparameters'
    CONFIG_HLS = 'hidden-layer-sizes'
    CONFIG_LR = 'learning-rate'
    CONFIG_ALPHA = 'alpha'
    CONFIG_SOLVER = 'solver'
    CONFIG_ACTIVATION = 'activation'
    CONFIG_MAX_ITER = 'max-iter'


class ModelConstants:
    MODELS_DIR = './src/models/'
    MODEL_BASE_NAME = 'classifier'


class LoggerConstants:
    APP_LOGGER_NAME = 'APP_LOGGER'
    APP_LOG_FILE_BASE_NAME = 'app'
    CLASSIFY_LOGGER_NAME = 'CLASSIFY_LOGGER'
    CLASSIFY_LOG_FILE_BASE_NAME = 'classify'
    TRAIN_LOGGER_NAME = 'TRAIN_LOGGER'
    TRAIN_LOG_FILE_BASE_NAME = 'train'
    LOGS_DIR = './logs/'


class IOConstants:
    READ = 'r'
    READ_BINARY = 'rb'
    WRITE = 'w'
    WRITE_BINARY = 'wb'