from AutoInsurance.config.configuration import ConfigurationManager
from AutoInsurance.components.user_app import RiskProfileModel 
from AutoInsurance import logger

STAGE_NAME = "User App stage"

class UserAppPipeline():
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        user_app_config = config.get_user_app_config()
        risk_profile_model = RiskProfileModel(config = user_app_config)
        return risk_profile_model


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = UserAppPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e