import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from AutoInsurance.entity.config_entity import RiskProfilesConfig
import joblib

class RiskProfiling:
    def __init__(self, config: RiskProfilesConfig):
        self.config = config
        self.params = config.params

    def load_data(self):
        return pd.read_csv(Path(self.config.potential_customers_with_predictions_data_path))

    def save_data(self, data: pd.DataFrame):
        data.to_csv(Path(self.config.risk_profiles_path), index=False)

    def normalize_data(self, data: pd.DataFrame):
        scaler = MinMaxScaler()
        data[['normalized_claim_probability', 'normalized_claim_amount']] = scaler.fit_transform(
            data[['claim_probability', 'claim_amount']]
        )
        joblib.dump(scaler, Path('artifacts/risk_profiles/minmax_scaler.pkl'))
        return data

    def segment_by_claim_probability(self, data: pd.DataFrame):
        quantiles = data['claim_probability'].quantile(self.params['claim_probability_thresholds'])
        data['risk_profile_probability'] = pd.cut(
            data['claim_probability'],
            bins=[0] + quantiles.tolist() + [1],
            labels=['Low', 'Medium', 'High']
        )
        return data

    def segment_by_claim_amount(self, data: pd.DataFrame):
        customers_with_claims = data[data['claim'] == 1].copy()
        quantiles = customers_with_claims['claim_amount'].quantile(self.params['claim_amount_thresholds'])
        customers_with_claims['risk_profile_cost'] = pd.cut(
            customers_with_claims['claim_amount'],
            bins=[0] + quantiles.tolist() + [customers_with_claims['claim_amount'].max()],
            labels=['Low', 'Medium', 'High']
        )
        data = data.merge(customers_with_claims[['risk_profile_cost']], left_index=True, right_index=True, how='left')
        if 'No Claim' not in data['risk_profile_cost'].cat.categories:
            data['risk_profile_cost'] = data['risk_profile_cost'].cat.add_categories('No Claim')
        data['risk_profile_cost'] = data['risk_profile_cost'].fillna('No Claim')
        return data

    def apply_dynamic_weighting(self, data: pd.DataFrame):
        weights_probability = self.params['weights_probability']
        weights_cost = self.params['weights_cost']

        data['weighted_probability_score'] = data.apply(
            lambda x: x['normalized_claim_probability'] * weights_probability[x['risk_profile_probability']], axis=1
        )
        data['weighted_cost_score'] = data.apply(
            lambda x: x['normalized_claim_amount'] * weights_cost[x['risk_profile_cost']], axis=1
        )
        return data

    def calculate_combined_risk_score(self, data: pd.DataFrame):
        data['dynamic_combined_risk_score'] = data['weighted_probability_score'] + data['weighted_cost_score']
        return data

    def segment_into_risk_groups(self, data: pd.DataFrame):
        quantiles = data['dynamic_combined_risk_score'].quantile(self.params['risk_score_thresholds'])
        data['risk_group'] = pd.cut(
            data['dynamic_combined_risk_score'],
            bins=[0] + quantiles.tolist() + [data['dynamic_combined_risk_score'].max()],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        return data

    def process_data(self):
        data = self.load_data()
        data = self.normalize_data(data)
        data = self.segment_by_claim_probability(data)
        data = self.segment_by_claim_amount(data)
        data = self.apply_dynamic_weighting(data)
        data = self.calculate_combined_risk_score(data)
        data = self.segment_into_risk_groups(data)
        self.save_data(data)
        print(data.head())
        return data
