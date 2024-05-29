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

@dataclass(frozen=True)
class ClassModelEvaluationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    class_model_path: Path
    all_params: dict
    class_metric_file_name: Path
    mlflow_uri: str

@dataclass(frozen=True)
class RegModelEvaluationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    reg_model_path: Path
    all_params: dict
    reg_metric_file_name: Path
    mlflow_uri: str

@dataclass(frozen=True)
class PredictionsConfig:
    root_dir: Path
    test_data_path: Path
    threshold_path: Path
    class_model_path: Path
    reg_model_path: Path
    potential_customers_data_path: Path
    potential_customers_with_predictions_data_path: Path

@dataclass(frozen=True)
class RiskProfilesConfig:
    root_dir: Path
    potential_customers_with_predictions_data_path: Path
    risk_profiles_path: Path
    params: dict

@dataclass(frozen=True)
class UserAppConfig:
    root_dir: Path
    risk_profiles_path: Path
    class_model_path: Path
    reg_model_path: Path
    test_data_path: Path
    scaler_path: Path
    params: dict