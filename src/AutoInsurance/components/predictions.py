import joblib 
import pandas as pd
from pathlib import Path
from AutoInsurance.utils.common import load_json
import numpy as np
from AutoInsurance.utils.common import logger
from AutoInsurance.config.configuration import PredictionsConfig

class Predictions:
    def __init__(self, config: PredictionsConfig):
        self.config = config

    def get_predictions(self):

        logger.info(f"class_model_path: {self.config.class_model_path}")
        logger.info(f"reg_model_path: {self.config.reg_model_path}")
        logger.info(f"test_data_path: {self.config.test_data_path}")
        logger.info(f"threshold_path: {self.config.threshold_path}")
        logger.info(f"potential_customers_data_path: {self.config.potential_customers_data_path}")


        class_model = joblib.load(Path(self.config.class_model_path))
        reg_model = joblib.load(Path(self.config.reg_model_path))

        test_data = pd.read_csv(Path(self.config.test_data_path))
        class_predictions_probs = class_model.predict_proba(test_data)[:, 1]

        metrics = load_json(Path(self.config.threshold_path))
        class_predictions = (class_predictions_probs >= metrics.optimal_threshold).astype(int)

        reg_predictions = reg_model.predict(test_data)
        reg_predictions = np.expm1(reg_predictions)

        potential_customers_data = pd.read_csv(Path(self.config.potential_customers_data_path))
        potential_customers_data['claim_probability'] = class_predictions_probs
        potential_customers_data['claim'] = class_predictions
        potential_customers_data['claim_amount'] = reg_predictions

        potential_customers_data.to_csv(Path(self.config.potential_customers_with_predictions_data_path), index = False)