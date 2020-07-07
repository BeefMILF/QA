from functools import partial

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from catalyst.data import BalanceClassSampler

from settings import SEED, BATCH_SIZE, TRAIN_SAMPLER_MODE


class BQDataset(Dataset):
    def __init__(self, data):
        super().__init__()

        if len(data) == 3:
            self.input_ids, self.attention_mask, self.targets = [torch.tensor(i).long() for i in data]
            self._item = lambda idx: {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'targets': self.targets[idx]
            }
        else:
            self.input_ids, self.attention_mask, self.token_type_ids, self.targets = [torch.tensor(i).long() for i in data]
            self._item = lambda idx: {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'token_type_ids': self.token_type_ids[idx],
                'targets': self.targets[idx]
            }

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self._item(idx)


samplers = {
    'train': {
        'balanced': partial(BalanceClassSampler, mode='upsampling'),
        'imbalanced': RandomSampler
    },
    'test': SequentialSampler,
    'val': SequentialSampler
}


def make_dataloader(features, mode='train'):

    dataset = BQDataset(features)

    # Set seeds for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    sampler = samplers[mode]
    if mode == 'train':
        sampler = sampler[TRAIN_SAMPLER_MODE]
        # pass labels
        sampler = sampler(features[-1])
    else:
        sampler = sampler(dataset)

    n_w = 4 if mode == 'train' else 1
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE, num_workers=n_w, drop_last=False)
    return dataloader