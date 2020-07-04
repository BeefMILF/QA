from typing import Mapping, Any

from sklearn.metrics import f1_score

from catalyst.dl import SupervisedRunner
from catalyst.dl import Callback, CallbackOrder

from bert_classifier import device


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
            'token_type_ids',
        ),
        device=device,
    )
    return runner