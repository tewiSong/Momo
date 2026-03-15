import yaml


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)
    return cfg

