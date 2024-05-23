from AutoInsurance.config.configuration import ConfigurationManager
from AutoInsurance.components.predictions import Predictions 
from AutoInsurance import logger

STAGE_NAME = "Predictions stage"

class PredictionsTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        predictions_config = config.get_predictions_config()
        predictions_config = Predictions(config = predictions_config)
        predictions_config.get_predictions()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PredictionsTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e