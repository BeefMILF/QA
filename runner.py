import torch

from misc import read_data, make_dataset, split_features, prepare_df
import settings
from settings import DATA, DATA_CLEAN_FLAG, NUM_EPOCHS, NUM_LABELS, LOG_DIR, GRAD_ACCUM_STEPS
from bert_tokenizer import BERTTextEncoder, tokenizer_kwargs
from bert_classifier import make_classifier, device
from dataloader import make_dataloader
import dl_runner
from catalyst.dl.callbacks import AccuracyCallback, SchedulerCallback, OptimizerCallback

# in case presplitted inds are needed or other variables
kwargs = {}


def make_data():
    # Loading data
    df_train = read_data(DATA['train'])
    df_dev = read_data(DATA['test'])
    print(f'Train df size: {df_train.shape}')
    print(f'Dev df size: {df_dev.shape}')

    # Clean data from trash
    if DATA_CLEAN_FLAG:
        df_train = prepare_df(df_train)
        df_dev = prepare_df(df_dev)

    # Can be added augmentation or etc.
    # ToDO

    # Pretrained tokenizer
    tokenizer = BERTTextEncoder()

    # String for printing data params
    string_log = '{n[0]} params size,  input ids: {n[1]}, attention masks: {n[2]}, token_type ids: {n[3]}, answers(targets): {n[4]}'

    # Encoding data
    train_features = make_dataset(df_train, tokenizer, tokenizer_kwargs)
    dev_features = make_dataset(df_dev, tokenizer, tokenizer_kwargs)
    print(string_log.format(n=['Train'] + [i.shape for i in train_features]))
    print(string_log.format(n=['Dev'] + [i.shape for i in dev_features]))

    # Split Dev into test/val
    # return inds in order to find fail-predicted samples
    test_features, val_features, inds = split_features(dev_features, p=0.7)
    print(string_log.format(n=['Validation'] + [i.shape for i in val_features]))
    print(string_log.format(n=['Test'] + [i.shape for i in test_features]))

    del dev_features

    kwargs = {'presplit_inds': inds}

    return train_features, val_features, test_features, kwargs


def make_loaders():
    global kwargs

    train_features, val_features, test_features, kwargs = make_data()
    loaders = {
        'train': make_dataloader(train_features, mode='train'),
        'valid': make_dataloader(val_features, mode='val'),
        'test': make_dataloader(val_features, mode='test')
    }
    for k, v in loaders.items():
        print(f'{k} loader size: {len(v)}')

    return loaders


def make_runner(model_kwargs):
    runner = dl_runner.make_runner()
    runner.train(**model_kwargs)


def run_model():
    loaders = make_loaders()
    model_kwargs = make_classifier()

    torch.cuda.empty_cache()
    print(f'Device: {device}')

    additional_kwargs = {
        'loaders': loaders,
        'callbacks': [
            AccuracyCallback(num_classes=NUM_LABELS),
            SchedulerCallback(mode='batch'),
            dl_runner.F1ScoreCallback(),
            OptimizerCallback(
                accumulation_steps=GRAD_ACCUM_STEPS,
                grad_clip_params={'func': 'clip_grad_value_', 'clip_value': 1}
            )
        ],
        'main_metric': 'accuracy01',
        'minimize_metric': False,
        'fp16': None,
        'logdir': LOG_DIR,
        'num_epochs': NUM_EPOCHS,
        'verbose': False,
        'load_best_on_end': True
    }
    model_kwargs = {**model_kwargs, **additional_kwargs}

    runner = make_runner(model_kwargs)


if __name__ == '__main__':
    settings.update_all('config_tmp.json')
    run_model()
