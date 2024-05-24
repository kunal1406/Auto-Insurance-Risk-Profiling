from AutoInsurance.config.configuration import ConfigurationManager
from AutoInsurance.components.risk_profiles import RiskProfiling 
from AutoInsurance import logger

STAGE_NAME = "Risk profiling stage"

class RiskProfilesPipeline():
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        risk_profiles_config = config.get_risk_profiles_config()
        predictions_config = RiskProfiling(config = risk_profiles_config)
        predictions_config.process_data()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = RiskProfilesPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e