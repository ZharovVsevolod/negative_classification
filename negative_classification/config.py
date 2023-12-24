from dataclasses import dataclass
from typing import Literal

@dataclass
class Dataset:
    path_to_data: str
    vocab_size: int
    path_to_bpe_dir: str
    chunk_lenght: int
    pad_value: int
    need_to_train_bpe: bool
    test_size_split: float

@dataclass
class Training:
    batch: int
    lr: float
    epochs: int

@dataclass
class Model:
    embedding_dim: int
    layers: int
    heads: int
    mlp_dim: int
    qkv_bias: bool
    dropout: float
    norm_type: Literal["postnorm", "prenorm"]
    num_output_classes: int

@dataclass
class Params:
    dataset: Dataset
    training: Training
    model: Model