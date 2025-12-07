from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:         # data ingestion config
    root_dir : Path
    source_url : str
    local_data_file : Path
    unzip_dir : Path


@dataclass(frozen=True)
class BaseModelConfig:           # base model config
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
class ModelTrainingConfig:       # model training config
    root_dir : Path
    trained_model_path : Path
    updated_base_model_path : Path
    training_data : Path
    params_epochs : int
    params_batch_size : int
    params_image_size : list
    params_learning_rate : float
    params_is_augmentation : bool

@dataclass(frozen=True)
class ModelEvaluationConfig:  # model evaluation configuration
    model_path : Path
    training_data : Path
    all_params : dict
    params_image_size : list
    params_batch_size : int
    params_epochs : int