from tqdm import tqdm
import pandas as pd
import numpy as np

from settings import SEED

import nltk
import string
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

stopWordsEng = stopwords.words('english')
pattern = re.compile(r'\b(' + r'|'.join(stopWordsEng) + r')\b\s*')


def read_data(data_path):
    # Loading data
    return pd.read_json(data_path, lines=True, orient='records')


def encode_data(tokenizer, questions, passages, kwargs):
    """Encode the question/passage pairs into features than can be fed to the model."""
    return tokenizer.batch_encode_plus(zip(questions, passages), max_length=kwargs['max_length'],
                                       padding='max_length', truncation='longest_first')


def make_dataset(df, tokenizer, kwargs):
    passages = df.passage.values
    questions = df.question.values
    answers = df.answer.values.astype(int)
    encoded_pair = encode_data(tokenizer, questions, passages, kwargs)

    if 'token_type_ids' in encoded_pair:
        return np.array(encoded_pair['input_ids']),    \
               np.array(encoded_pair['attention_mask']),\
               np.array(encoded_pair['token_type_ids']), \
               answers
    else:
        return np.array(encoded_pair['input_ids']),    \
               np.array(encoded_pair['attention_mask']),\
               answers


def split_features(features, p=0.5, seed=SEED):
    n = len(features[-1])
    # for reproducability
    np.random.seed(SEED)
    inds = np.random.choice(2, n, p=[1 - p, p]).astype(bool)
    features1 = [feature[~inds] for feature in features if feature is not None]
    features2 = [feature[inds] for feature in features if feature is not None]
    return features1, features2, inds


def text_prepare(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[{}]'.format(string.punctuation), '', text)
    text = re.sub('[^A-Za-z\. ]', '', text)
    text = pattern.sub('', text)
    return text


def prepare_df(df):
    # df.question = df.question.apply(text_prepare)
    df.passage = df.passage.apply(text_prepare)
    return df
