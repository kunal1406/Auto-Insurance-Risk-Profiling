from AutoInsurance.config.configuration import ConfigurationManager
from AutoInsurance.components.model_evaluation import ClassModelEvaluation, RegModelEvaluation
from AutoInsurance import logger

STAGE_NAME = "Model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()

        class_model_evaluation_config = config.get_class_model_evaluation_config()
        class_model_evaluation_config = ClassModelEvaluation(config=class_model_evaluation_config)
        class_model_evaluation_config.log_into_mlflow()
        logger.info("logged into mlflow for classification")
        reg_model_evaluation_config = config.get_reg_model_evaluation_config()
        reg_model_evaluation_config = RegModelEvaluation(config=reg_model_evaluation_config)
        reg_model_evaluation_config.log_into_mlflow()
        logger.info("logged into mlflow for regression")

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e