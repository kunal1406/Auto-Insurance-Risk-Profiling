from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class ClassModelTrainerConfig:
    root_dir: Path
    train_data_class_path: Path
    test_data_path: Path
    model_class_name: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_leaf: int
    max_features: float

@dataclass(frozen=True)
class RegModelTrainerConfig:
    root_dir: Path
    train_data_reg_path: Path
    test_data_path: Path
    model_reg_name: str
    learning_rate: float
    max_depth: int
    max_features: float
    min_samples_leaf: int
    n_estimators: int