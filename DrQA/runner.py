import torch
from torch import nn
from torch import optim

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from DrQA import dataloader
from DrQA import DrQA_model
from DrQA import dl_runner

PAR_EMBEDDING = 637
EMBEDDING_DIM = 300

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def default_config():
    return {
        'batch_size': 32,
        'n_layers': 2,
        'hidden_size': 512,
        'embedding_dim': EMBEDDING_DIM,
        'paragraph_embedding': PAR_EMBEDDING,
        'bidirectional': True,
        'dropout': 0.25,
        'dropout_fc': 0.15,
        'encode_q': False,
        'pooler': {
            'mode': False,
            'kernel_size': 4,
            'stride': 2,
        },
        'pretrained': None,
        'train': {
            'optim': {
                'lr': 1e-3,
                'weight_decay': 0
            },
            'scheduler': {
                # ReduceLROnPlateau
                # 'factor': 0.5,
                # 'patience': 1,
                # 'min_lr': 1e-4
                'step_size': 1,
                'gamma': 0.3,
            },
            'n_epochs': 3,
        }
    }


def run_model(config=None, glove=False, device=device):
    if config is None:
        config = default_config()

    dl = dataloader.DataLoader(glove)
    print('DataLoader done...')
    config['pretrained'] = dl.word_vocab.vectors

    loaders = {
        'train': dataloader.bucket(dl.train_data, config['batch_size'], mode='train', device=device),
        'test': dataloader.bucket(dl.dev_data, config['batch_size'], mode='test', device=device)
    }
    print('Model configuration...')
    model = DrQA_model.DrQA(config, dl, device)

    criterion = nn.CrossEntropyLoss()
    model = dl_runner.Model(model, criterion, config, loaders)
    logger = TensorBoardLogger('tb_logs', name='drqa')
    trainer = Trainer(
        gpus=1,
        logger=logger,
        max_epochs=config['train']['n_epochs'],
        accumulate_grad_batches=3,
        gradient_clip_val=0.4
    )
    print('Model training')
    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    run_model()
