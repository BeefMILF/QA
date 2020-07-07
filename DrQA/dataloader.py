import warnings
warnings.filterwarnings("ignore")

import torch

import jsonlines

from tqdm import tqdm
from collections import defaultdict

import spacy
from torchtext.vocab import GloVe, FastText
from torchtext.data import Example, Field, Dataset, NestedField, BucketIterator


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DataLoader:
    def __init__(self, glove=True, device=device):
        self.device = device

        nlp = spacy.load("en_core_web_sm")

        char_nesting = Field(batch_first=True, tokenize=list, lower=True)
        char = NestedField(char_nesting, init_token="<sos>", eos_token="<eos>", tokenize="spacy")
        word = Field(init_token="<sos>", eos_token="<eos>", lower=True, tokenize="spacy")
        label = Field(sequential=False, is_target=True, use_vocab=False)

        self.fields = [("question_char", char), ("question_word", word),
                       ("context_char", char), ("context_word", word),
                       ("answer", label)]

        self.dict_fields = {"question": [("question_char", char), ("question_word", word)],
                            "context": [("context_char", char), ("context_word", word)],
                            "answer": ("answer", label)}

        self.train_data = self._get_data("../data/train.jsonl")
        self.dev_data = self._get_data("../data/dev.jsonl")

        char.build_vocab(self.train_data)
        if glove:
            word.build_vocab(self.train_data, vectors=GloVe(name="6B", dim=100))
        else:
            word.build_vocab(self.train_data, vectors=FastText(language='en', max_vectors=30000))

        self.char_vocab = char.vocab
        self.word_vocab = word.vocab

        pos = []
        ner = []

        ind2pos = []
        ind2ner = []

        for data in tqdm(self.train_data):
            doc = nlp(' '.join(data.question_word + data.context_word))

            # t - token
            pos.extend([t.pos_ for t in doc])
            ner.extend([t.label_ for t in doc.ents])

            ind2pos.extend([[self.word_vocab.stoi[str(t)], t.pos_] for t in doc])
            ind2ner.extend([[self.word_vocab.stoi[str(t)], t.label_] for t in doc.ents])

        self.pos_voc = {tag: i for i, tag in enumerate(set(pos))}
        self.ner_voc = {tag: i + 1 for i, tag in enumerate(set(ner))}
        self.ner_voc['None'] = 0

        # default values, used in DrQA model
        self.ind2pos = defaultdict(lambda: self.pos_voc['X']) # returns 14
        self.ind2ner = defaultdict(lambda: self.ner_voc['None']) # returns 0

        self.ind2pos.update({tag[0] : self.pos_voc[tag[1]] for tag in ind2pos})
        self.ind2ner.update({tag[0] : self.ner_voc[tag[1]] for tag in ind2ner})

    def _get_data(self, file):
        examples = []

        with jsonlines.open(file) as json_lines:
            for e in json_lines:
                example = {"context": e["title"] + ' ' + e["passage"],
                           "question": e["question"],
                           "answer": e["answer"]}
                examples.append(Example.fromdict(example, fields=self.dict_fields))

        return Dataset(examples, self.fields)

    # @staticmethod
    # def save_vocab(vocab, path):
    #     with open(path, 'w+') as f:
    #         for token, index in vocab.stoi.items():
    #             f.write(f'{index}\t{token}')
    #
    # @staticmethod
    # def read_vocab(path):
    #     vocab = dict()
    #     with open(path, 'r') as f:
    #         for line in f:
    #             index, token = line.split('\t')
    #             vocab[token] = int(index)
    #     return vocab


def bucket(data, batch_size=64, mode='train', device=device):
    shuffle = True if mode == 'train' else False
    return BucketIterator(data, batch_size=batch_size, sort_key=lambda x: len(x.passage_word),
                          device=device, shuffle=shuffle)
