from typing import Mapping
import json


def load_config(config_path='config_tmp.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(data, config_path='config_tmp.json', indent=1):
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=1)


def custom_change(kwargs: Mapping, config_path='config.json', dist_path='config_tmp.json'):
    """ Primarily for changing hyperparameters and other model modifications """
    data = load_config(config_path)
    data.update(kwargs)
    save_config(data, dist_path)

