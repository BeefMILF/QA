import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from transformers import AutoConfig, AutoModel
from transformers import get_linear_schedule_with_warmup
from catalyst.contrib.nn.optimizers import RAdam
from catalyst.contrib.nn import OneCycleLRWithWarmup

from catalyst.utils import set_global_seed, prepare_cudnn

from settings import SEED, PRETRAINED_MODEL_NAME, NUM_LABELS, LR, LR_RANGE, NUM_EPOCHS, WARMUP_STEPS

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BertForSequenceClassification(nn.Module):
    def __init__(self, model_name: str = PRETRAINED_MODEL_NAME, num_cls=NUM_LABELS):
        """
        Args:
            model_name (str): HuggingFace model name.
                See transformers/modeling_auto.py
            num_cls (int): the number of class labels
                in the classification task
        """
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_cls)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, num_cls)
        # self.init_weights(self.classifier)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._frozen = False

    def n_trainable(self):
        return sum([params.numel() for name, params in self.named_parameters() if params.requires_grad])

    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def unfreeze_encoder(self):
        """ unfreeze bert layers """
        if self._frozen:
            print('Bert model fine-tuning')
            for p in self.bert.parameters():
                p.requires_grad = True
            self._frozen = False

    def freeze_encoder(self):
        """ freeze bert layers """
        for p in self.bert.parameters():
            p.requires_grad = False
        self._frozen = True

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        assert attention_mask is not None, "attention mask is none"

        # we only need hidden state, not transformer
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        hidden_state = output[0]  # (bs, seq_len, dim)
        # take embeddings from the [CLS] token
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_cls)
        return logits

    def configure_loss(self):
        """ Initialize the loss function """
        criterion = nn.CrossEntropyLoss()
        return criterion

    def configure_optimizers(self, epoch=0, lr=LR):
        """ Initialize parameters for optimizer """
        # parameters = [
        #     {'params': self.classifier.parameters(), 'lr': 2e-4, 'weight_decay': 1e-5}
        # ]
        # if epoch:
        #    parameters.append({'params': self.bert.parameters(), 'lr': lr})
        # else:
        #     self.freeze_encoder()
        #     print(f'First epoch, freezing encoder\nTrainable parameters: {self.n_trainable()}')
        parameters = {'params': self.parameters(), 'lr': LR, 'weight_decay': 0}
        optimizer = RAdam(**parameters)
        return optimizer

    def configure_scheduler(self, optimizer):
        """ Initialize cyclic scheduler for lr modification """
        return OneCycleLRWithWarmup(optimizer, num_steps=NUM_EPOCHS, lr_range=LR_RANGE, warmup_steps=WARMUP_STEPS,
                                    momentum_range=(0.9, 0.92))


def make_classifier():
    # Set seeds for reproducibility
    set_global_seed(SEED)
    prepare_cudnn(deterministic=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = BertForSequenceClassification(PRETRAINED_MODEL_NAME, NUM_LABELS)
    model.to(device)
    print(f'Loaded model: {PRETRAINED_MODEL_NAME}')
    print(f'Trainable parameters: {model.n_trainable()}')

    criterion = model.configure_loss()
    optimizer = model.configure_optimizers(1)
    scheduler = model.configure_scheduler(optimizer)

    return {
        'criterion': criterion,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'model': model,
        # 'epoch': 0
    }

