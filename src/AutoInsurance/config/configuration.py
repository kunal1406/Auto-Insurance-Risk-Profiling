from AutoInsurance.constants import *
from AutoInsurance.utils.common import read_yaml, create_directories 
from AutoInsurance.entity.config_entity import (DataIngestionConfig, DataValidationConfig, DataTransformationConfig, 
                                                ClassModelTrainerConfig, RegModelTrainerConfig, ClassModelEvaluationConfig,
                                                  RegModelEvaluationConfig, PredictionsConfig, RiskProfilesConfig, UserAppConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir= config.root_dir,
            train_data_path= config.train_data_path,
            test_data_path= config.test_data_path,
        )

        return data_transformation_config
    
    def get_class_model_trainer_config(self) -> ClassModelTrainerConfig:
        config = self.config.class_model_trainer
        params = self.params.GradientBoostingClassifier

        create_directories([config.root_dir])

        class_model_trainer_config = ClassModelTrainerConfig(
            root_dir= config.root_dir,
            train_data_class_path= config.train_data_class_path,
            test_data_path= config.test_data_path,
            model_class_name= config.model_class_name,
            n_estimators= params.n_estimators,
            learning_rate= params.learning_rate,
            max_depth= params.max_depth,
            min_samples_leaf= params.min_samples_leaf,
            max_features= params.max_features,
        )

        return class_model_trainer_config
    
    def get_reg_model_trainer_config(self) -> RegModelTrainerConfig:
        config = self.config.reg_model_trainer
        params = self.params.GradientBoostingRegressor

        create_directories([config.root_dir])    

        reg_model_trainer_config = RegModelTrainerConfig(
            root_dir= config.root_dir,
            train_data_reg_path= config.train_data_reg_path,
            test_data_path= config.test_data_path,
            model_reg_name= config.model_reg_name,
            n_estimators= params.n_estimators,
            learning_rate= params.learning_rate,
            max_depth= params.max_depth,
            min_samples_leaf= params.min_samples_leaf,
            max_features= params.max_features,
        )

        return reg_model_trainer_config  

    def get_class_model_evaluation_config(self) -> ClassModelEvaluationConfig:
        config = self.config.class_model_evaluation
        params = self.params.GradientBoostingClassifier

        create_directories([config.root_dir])

        class_model_evaluation_config = ClassModelEvaluationConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_class_path,
            test_data_path= config.test_data_path,
            class_model_path = config.class_model_path,
            all_params= params,
            class_metric_file_name = config.class_metric_file_name,
            mlflow_uri = "https://dagshub.com/kunal1406/Auto-Insurance-Risk-Profiling.mlflow"
        )

        return class_model_evaluation_config
    
    def get_reg_model_evaluation_config(self) -> RegModelEvaluationConfig:
        config = self.config.reg_model_evaluation
        params = self.params.GradientBoostingRegressor

        create_directories([config.root_dir])

        reg_model_evaluation_config = RegModelEvaluationConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_reg_path,
            test_data_path= config.test_data_path,
            reg_model_path = config.reg_model_path,
            all_params= params,
            reg_metric_file_name = config.reg_metric_file_name,
            mlflow_uri = "https://dagshub.com/kunal1406/Auto-Insurance-Risk-Profiling.mlflow"
        )

        return reg_model_evaluation_config   
    
    def get_predictions_config(self) -> PredictionsConfig:
        config = self.config.predictions

        create_directories([config.root_dir])
        predictions_config =  PredictionsConfig(
            root_dir = Path(config.root_dir),
            threshold_path = Path(config.threshold_path),
            test_data_path = Path(config.test_data_path),
            class_model_path = Path(config.class_model_path),
            reg_model_path = Path(config.reg_model_path),
            potential_customers_data_path = Path(config.potential_customers_data_path),
            potential_customers_with_predictions_data_path = Path(config.potential_customers_with_predictions_data_path)
        )
    
        return predictions_config    
    
    def get_risk_profiles_config(self) -> RiskProfilesConfig:
        config = self.config.risk_profiles
        params = self.params.RiskProfiles

        create_directories([config.root_dir])

        risk_profiles_config = RiskProfilesConfig(
            root_dir= Path(config.root_dir),
            potential_customers_with_predictions_data_path= Path(config.potential_customers_with_predictions_data_path),
            risk_profiles_path= Path(config.risk_profiles_path),
            params= params
        )

        return risk_profiles_config
    
    def get_user_app_config(self) -> UserAppConfig:
        config = self.config.user_app
        params = self.params.RiskProfiles

        create_directories([config.root_dir])

        user_app_config = UserAppConfig(
            root_dir= Path(config.root_dir),
            test_data_path= Path(config.test_data_path),
            risk_profiles_path= Path(config.risk_profiles_path),
            class_model_path= Path(config.class_model_path),
            reg_model_path= Path(config.reg_model_path),
            scaler_path= Path(config.scaler_path),
            params= params
        )

        return user_app_config 