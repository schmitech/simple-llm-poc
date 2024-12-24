import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    max_length: int
    batch_size: int
    device: str
    quantization: dict

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config['model'])

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)