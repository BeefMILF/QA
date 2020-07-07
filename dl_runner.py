from typing import Mapping, Any

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from catalyst.dl import SupervisedRunner
from catalyst.dl import Callback, CallbackOrder
from catalyst.dl.callbacks import CheckpointCallback, InferCallback

from bert_classifier import device

from settings import LOG_DIR


class F1ScoreCallback(Callback):
    def __init__(
            self,
            input_key: str = 'targets',
            output_key: str = 'logits',
            activation: str = 'Sigmoid',
            prefix: str = "f1_score",
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix

        super().__init__(CallbackOrder.Metric)

    def on_batch_end(self, state):
        y_true = state.input[self.input_key].detach().cpu().numpy()
        y_preds = state.output[self.output_key].detach().cpu().numpy().argmax(1)

        score = f1_score(y_true, y_preds)

        state.batch_metrics.update({self.prefix: score})


def make_runner():
    runner = SupervisedRunner(
        input_key=(
            'input_ids',
            'attention_mask',
            # 'token_type_ids',
        ),
        device=device,
    )
    return runner


def calc_accuracy(y_preds, y_true):
    acc = accuracy_score(y_true, y_preds)
    return acc


def calc_accuracy_per_cls(y_preds, y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    acc1, acc2 = tn / (tn + fn), tp / (tp + fp)
    return acc1, acc2


def calc_f1_score(y_preds, y_true):
    f1 = f1_score(y_true, y_preds)
    return f1


def make_predictions(runner, model, loader, y_true):
    runner.infer(
        model=model,
        loaders=loader,
        callbacks=[
            CheckpointCallback(
                resume=f"{LOG_DIR}/checkpoints/best.pth"
            ),
            InferCallback(),
        ],
        verbose=True
    )

    y_preds = runner.callbacks[0].predictions['logits'].argmax(1)

    acc = calc_accuracy(y_preds, y_true)
    acc1, acc2 = calc_accuracy_per_cls(y_preds, y_true)
    f1 = calc_f1_score(y_preds, y_true)

    return {'acc': acc,
            'acc1': acc1,
            'acc2': acc2,
            'f1': f1}
