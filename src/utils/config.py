from dataclasses import dataclass
from typing import Union

from sklearn.base import ClassifierMixin
from torch.nn import Module

from src.interfaces.base_preprocessor import BasePreProcessor
from src.external.musicnn import Musicnn


@dataclass
class Config:
    preprocessor: BasePreProcessor = None
    model: Union[Module, ClassifierMixin] = Musicnn()
    n_epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    model_filename_path: str = "models"
    data_path: str = 'data'
    log_step: int = 100
    sr: int = 16000
    input_length: int = 3 * sr
    dataset_split_path: str = "split"
    dataset_name: str = "mtat"
    logs_path: str = "logs"
