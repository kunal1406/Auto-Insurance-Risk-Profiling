from AutoInsurance.config.configuration import ConfigurationManager
from AutoInsurance.components.user_app import RiskProfileModel 
from AutoInsurance import logger
import pandas as pd
from pathlib import Path

STAGE_NAME = "User App stage"

class UserAppPipeline():
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        user_app_config = config.get_user_app_config()
        risk_profile_model = RiskProfileModel(config = user_app_config)
        risk_profiles_df = pd.read_csv(Path(risk_profile_model.config.risk_profiles_path), dtype= {'agecat':'object', 'veh_age': 'object'} )
        test = pd.read_csv(Path('artifacts/data_transformation/Processed_test_data.csv'))
        return risk_profile_model, risk_profiles_df, test


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = UserAppPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e