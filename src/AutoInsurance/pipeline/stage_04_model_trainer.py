from AutoInsurance.config.configuration import ConfigurationManager
from AutoInsurance.components.model_trainer import (ClassModelTrainer, RegModelTrainer)
from AutoInsurance import logger

STAGE_NAME = "Model Training stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        class_model_trainer_config = config.get_class_model_trainer_config()
        class_model_trainer_config = ClassModelTrainer(config = class_model_trainer_config)
        class_model_trainer_config.train_model()
        reg_model_trainer_config = config.get_reg_model_trainer_config()
        reg_model_trainer_config = RegModelTrainer(config = reg_model_trainer_config)
        reg_model_trainer_config.train_model()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.error(e)
        logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<\n\nx==========x")