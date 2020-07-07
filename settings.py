import os
from config import load_config


config_path = 'config_tmp.json' if os.path.exists('config_tmp.json') else 'config.json'
config_kwargs = load_config(config_path)


DATA_PATH = config_kwargs.get('data_path', './data')

DATA = {
    'train': config_kwargs.get('train', os.path.join(DATA_PATH, 'train.jsonl')),
    'test': config_kwargs.get('test', os.path.join(DATA_PATH, 'dev.jsonl'))
}

DATA_CLEAN_FLAG = config_kwargs.get('data_clean_flag', True)

SEED = config_kwargs.get('seed', 3)

MAX_QUESTION_LENGTH = config_kwargs.get('max_q_len', None)
MAX_PASSAGE_LENGTH = config_kwargs.get('max_p_len', None)
MAX_SEQ_LENGTH = config_kwargs.get('max_seq_len', 512)

PRETRAINED_MODEL_NAME = config_kwargs.get('pretrained_model_name', 'google/bert_uncased_L-4_H-256_A-4')

BATCH_SIZE = config_kwargs.get('batch_size', 8)

TRAIN_SAMPLER_MODE = config_kwargs.get('train_sampler_mode', 'balanced')

NUM_LABELS = config_kwargs.get('num_labels', 2)

LR = config_kwargs.get('lr', 5e-5)
LR_RANGE = config_kwargs.get('lr_range', (1e-5, 2e-6))

GRAD_ACCUM_STEPS = config_kwargs.get('grad_accum_steps', 4)

NUM_EPOCHS = config_kwargs.get('num_epochs', 8)

WARMUP_STEPS = config_kwargs.get('warmup_steps', NUM_EPOCHS // 2)

LOG_DIR = config_kwargs.get('logdir', f'logs/{PRETRAINED_MODEL_NAME}')


def update_all(config_path=config_path):
    config_kwargs = load_config(config_path)

    global DATA_PATH, DATA, DATA_CLEAN_FLAG, SEED, MAX_QUESTION_LENGTH, MAX_PASSAGE_LENGTH, MAX_SEQ_LENGTH, \
        PRETRAINED_MODEL_NAME, BATCH_SIZE, TRAIN_SAMPLER_MODE, NUM_LABELS, LR, LR_RANGE, GRAD_ACCUM_STEPS, NUM_EPOCHS, LOG_DIR

    DATA_PATH = config_kwargs.get('data_path', './data')

    DATA = {
        'train': config_kwargs.get('train', os.path.join(DATA_PATH, 'train.jsonl')),
        'test': config_kwargs.get('test', os.path.join(DATA_PATH, 'dev.jsonl'))
    }

    DATA_CLEAN_FLAG = config_kwargs.get('data_clean_flag', True)

    SEED = config_kwargs.get('seed', 3)

    MAX_QUESTION_LENGTH = config_kwargs.get('max_q_len', None)
    MAX_PASSAGE_LENGTH = config_kwargs.get('max_p_len', None)
    MAX_SEQ_LENGTH = config_kwargs.get('max_seq_len', 512)

    PRETRAINED_MODEL_NAME = config_kwargs.get('pretrained_model_name', 'google/bert_uncased_L-4_H-256_A-4')

    BATCH_SIZE = config_kwargs.get('batch_size', 8)

    TRAIN_SAMPLER_MODE = config_kwargs.get('train_sampler_mode', 'balanced')

    NUM_LABELS = config_kwargs.get('num_labels', 2)

    LR = config_kwargs.get('lr', 5e-5)
    LR_RANGE = config_kwargs.get('lr_range', (1e-5, 2e-6))

    GRAD_ACCUM_STEPS = config_kwargs.get('grad_accum_steps', 4)

    NUM_EPOCHS = config_kwargs.get('num_epochs', 8)

    WARMUP_STEPS = config_kwargs.get('warmup_steps', NUM_EPOCHS // 2)

    LOG_DIR = config_kwargs.get('logdir', f'logs/{PRETRAINED_MODEL_NAME}')