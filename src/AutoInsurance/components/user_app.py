import pandas as pd
import joblib
from AutoInsurance.entity.config_entity import UserAppConfig
import numpy as np
from pathlib import Path


class RiskProfileModel:
    # def __init__(self, config: UserAppConfig):
    #     self.config = config
    #     self.params = config.params
    def __init__(self, config: UserAppConfig ):
        self.config = config
        self.params = config.params
        # self.risk_profiles_df = pd.read_csv(risk_profile_path)

    def transform_user_data_to_df(self, user_data, columns, dtypes):
        data = {col: np.zeros(1, dtype=dt) if dt == 'float64' else np.zeros(1, dtype=bool) for col, dt in dtypes.items()}
        df = pd.DataFrame(data)

        if 'credit_score' in user_data:
            df['credit_score'] = user_data['credit_score']
        if 'traffic_index' in user_data:
            df['traffic_index'] = user_data['traffic_index']
        if 'veh_value' in user_data:
            df['veh_value'] = user_data['veh_value']

        for key, value in user_data.items():
            if key in ['gender', 'area', 'veh_body', 'agecat', 'veh_age']:
                column_name = f'{key}_{value}'
                if column_name in df.columns:
                    df[column_name] = True

        return df

    def predict(self, user_data, columns, dtypes):
        df = self.transform_user_data_to_df(user_data, columns, dtypes)
        class_predictions_probs = joblib.load(Path(self.config.class_model_path)).predict_proba(df)[:, 1]
        reg_predictions = joblib.load(Path(self.config.reg_model_path)).predict(df)
        reg_predictions = np.expm1(reg_predictions)

        claim_likelihood = class_predictions_probs[0]
        claim_amount = reg_predictions[0]

        return claim_likelihood, claim_amount

    def normalize_predictions(self, claim_likelihood, claim_amount):

        features = pd.DataFrame([[claim_likelihood, claim_amount]], columns=['claim_probability', 'claim_amount'])
        norm_values = joblib.load(Path(self.config.scaler_path)).transform(features)
        normalized_claim_likelihood = norm_values[0, 0]
        normalized_claim_amount = norm_values[0, 1]

        return normalized_claim_likelihood, normalized_claim_amount

    def classify_risk(self, normalized_claim_likelihood, normalized_claim_amount, claim_likelihood, claim_amount):
        claim_probability_thresholds = [0.2, 0.45]
        claim_amount_thresholds = [0.2, 0.45]
        weights_probability = {'Low': 0.4, 'Medium': 0.5, 'High': 0.6}
        weights_cost = {'No Claim': 0.3, 'Low': 0.4, 'Medium': 0.5, 'High': 0.6}
        risk_score_thresholds = [0.2, 0.45]

        risk_profiles_df = pd.read_csv(Path(self.config.risk_profiles_path))

        quantiles_prob = risk_profiles_df['claim_probability'].quantile(claim_probability_thresholds)
        if normalized_claim_likelihood <= quantiles_prob.iloc[0]:
            risk_profile_probability = 'Low'
        elif normalized_claim_likelihood <= quantiles_prob.iloc[1]:
            risk_profile_probability = 'Medium'
        else:
            risk_profile_probability = 'High'

        if claim_amount == 0:
            risk_profile_cost = 'No Claim'
        else:
            quantiles_cost = risk_profiles_df.loc[risk_profiles_df['claim_amount'] > 0, 'claim_amount'].quantile(claim_amount_thresholds)
            if normalized_claim_amount <= quantiles_cost.iloc[0]:
                risk_profile_cost = 'Low'
            elif normalized_claim_amount <= quantiles_cost.iloc[1]:
                risk_profile_cost = 'Medium'
            else:
                risk_profile_cost = 'High'

        weighted_probability_score = normalized_claim_likelihood * weights_probability[risk_profile_probability]
        weighted_cost_score = normalized_claim_amount * weights_cost[risk_profile_cost]
        dynamic_combined_risk_score = weighted_probability_score + weighted_cost_score

        quantiles_risk = risk_profiles_df['dynamic_combined_risk_score'].quantile(risk_score_thresholds)
        if dynamic_combined_risk_score <= quantiles_risk.iloc[0]:
            risk_group = 'Low Risk'
        elif dynamic_combined_risk_score <= quantiles_risk.iloc[1]:
            risk_group = 'Medium Risk'
        else:
            risk_group = 'High Risk'

        return {
            'claim_likelihood': claim_likelihood,
            'claim_amount': claim_amount,
            'normalized_claim_likelihood': normalized_claim_likelihood,
            'normalized_claim_amount': normalized_claim_amount,
            'risk_profile_probability': risk_profile_probability,
            'risk_profile_cost': risk_profile_cost,
            'dynamic_combined_risk_score': dynamic_combined_risk_score,
            'risk_group': risk_group
        }