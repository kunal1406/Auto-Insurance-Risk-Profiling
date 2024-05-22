from AutoInsurance.config.configuration import ConfigurationManager
from AutoInsurance.components.data_transformation import DataTransformation
from AutoInsurance import logger
from pathlib import Path

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                logger.info(f"checkpoint 1")
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                logger.info(f"checkpoint 2 {data_transformation}")
                data_transformation.load_and_transform_data()
            else:
                raise Exception("Your data schema is invalid")
            
        except Exception as e:
            print(e)


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e