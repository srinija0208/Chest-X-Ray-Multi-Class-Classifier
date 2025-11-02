from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir : Path
    source_url : str
    local_data_file : Path
    unzip_dir : Path


@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_learning_rate: float
    params_image_size: list
    params_classes: int
    params_include_top: bool
    params_weights: str
    params_epochs: int
    params_batch_size: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir : Path
    trained_model_path : Path
    updated_base_model_path : Path
    training_data : Path
    params_epochs : int
    params_batch_size : int
    params_image_size : list
    params_learning_rate : float
    params_is_augmentation : bool