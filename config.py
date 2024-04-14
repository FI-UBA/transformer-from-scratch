import yaml
from pathlib import Path


def get_config():
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    return config


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['dataset']['name']}_{config['training']['model_folder']}"
    model_filename = f"{config['training']['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['dataset']['name']}_{config['training']['model_folder']}"
    model_filename = f"{config['training']['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
