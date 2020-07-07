import torch
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy, F1


class Model(LightningModule):
    def __init__(self, model, criterion, config: dict, loaders: dict):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.config = config
        self.loaders = loaders
        self.metrics = {'acc': Accuracy(),
                        'f1': F1()}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), **self.config['train']['optim'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, **self.config['train']['scheduler'])
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.config['train']['scheduler'])
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        target = batch.answer
        out = self(batch)
        pred = torch.argmax(F.softmax(out.data), dim=-1)

        loss = self.criterion(out, target)
        f1 = self.metrics['f1'](pred, target)
        acc = self.metrics['acc'](pred, target)

        return {
            'loss': loss,
            'f1': f1,
            'acc': acc,
            'log': {
                'train_loss': loss,
                'train_f1': f1,
                'train_acc': acc
            }
        }

    def test_step(self, batch, batch_idx):
        target = batch.answer
        out = self(batch)
        pred = torch.argmax(F.softmax(out.data), dim=-1)

        loss = self.criterion(out, target)
        f1 = self.metrics['f1'](pred, target)
        acc = self.metrics['acc'](pred, target)

        return {
            'loss': loss,
            'f1': f1,
            'acc': acc,
            'log': {
                'test_loss': loss,
                'test_f1': f1,
                'test_acc': acc
            }
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['f1'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        return {
            'avg_train_loss': avg_loss,
            'avg_train_f1': avg_f1,
            'avg_train_acc': avg_acc,
            'log': {
                'train_loss': avg_loss,
                'train_f1': avg_f1,
                'train_acc': avg_acc
            }
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['f1'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        return {
            'avg_test_loss': avg_loss,
            'avg_test_f1': avg_f1,
            'avg_test_acc': avg_acc,
            'log': {
                'test_loss': avg_loss,
                'test_f1': avg_f1,
                'test_acc': avg_acc
            }
        }

    def train_dataloader(self):
        return self.loaders['train']

    def test_dataloader(self):
        return self.loaders['test']