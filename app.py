import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class RiskProfileModel:
    def __init__(self, model_paths, scaler_path, risk_profile_path):
        self.class_model = joblib.load(Path(model_paths['class_model']))
        self.reg_model = joblib.load(Path(model_paths['reg_model']))
        self.scaler = joblib.load(Path(scaler_path))
        self.risk_profiles_df = pd.read_csv(risk_profile_path)
    
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
        class_predictions_probs = self.class_model.predict_proba(df)[:, 1]
        reg_predictions = self.reg_model.predict(df)
        reg_predictions = np.expm1(reg_predictions)
        
        claim_likelihood = class_predictions_probs[0]
        claim_amount = reg_predictions[0]
        
        return claim_likelihood, claim_amount
    
    def normalize_predictions(self, claim_likelihood, claim_amount):

        features = pd.DataFrame([[claim_likelihood, claim_amount]], columns=['claim_probability', 'claim_amount'])
        norm_values = self.scaler.transform(features)
        normalized_claim_likelihood = norm_values[0, 0]
        normalized_claim_amount = norm_values[0, 1]
        
        return normalized_claim_likelihood, normalized_claim_amount
    
    def classify_risk(self, normalized_claim_likelihood, normalized_claim_amount, claim_amount):
        claim_probability_thresholds = [0.2, 0.45]
        claim_amount_thresholds = [0.2, 0.45]
        weights_probability = {'Low': 0.4, 'Medium': 0.5, 'High': 0.6}
        weights_cost = {'No Claim': 0.3, 'Low': 0.4, 'Medium': 0.5, 'High': 0.6}
        risk_score_thresholds = [0.2, 0.45]
        
        quantiles_prob = self.risk_profiles_df['claim_probability'].quantile(claim_probability_thresholds)
        if normalized_claim_likelihood <= quantiles_prob.iloc[0]:
            risk_profile_probability = 'Low'
        elif normalized_claim_likelihood <= quantiles_prob.iloc[1]:
            risk_profile_probability = 'Medium'
        else:
            risk_profile_probability = 'High'
        
        if claim_amount == 0:
            risk_profile_cost = 'No Claim'
        else:
            quantiles_cost = self.risk_profiles_df.loc[self.risk_profiles_df['claim_amount'] > 0, 'claim_amount'].quantile(claim_amount_thresholds)
            if normalized_claim_amount <= quantiles_cost.iloc[0]:
                risk_profile_cost = 'Low'
            elif normalized_claim_amount <= quantiles_cost.iloc[1]:
                risk_profile_cost = 'Medium'
            else:
                risk_profile_cost = 'High'
        
        weighted_probability_score = normalized_claim_likelihood * weights_probability[risk_profile_probability]
        weighted_cost_score = normalized_claim_amount * weights_cost[risk_profile_cost]
        dynamic_combined_risk_score = weighted_probability_score + weighted_cost_score
        
        quantiles_risk = self.risk_profiles_df['dynamic_combined_risk_score'].quantile(risk_score_thresholds)
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


model_paths = {
    'class_model': "artifacts/model_trainer/class_model.joblib",
    'reg_model': 'artifacts/model_trainer/reg_model.joblib'
}
scaler_path = 'artifacts/risk_profiles/minmax_scaler.pkl'
risk_profile_path = 'artifacts/risk_profiles/risk_profiles.csv'

risk_profile_model = RiskProfileModel(model_paths, scaler_path, risk_profile_path)

test = pd.read_csv('artifacts/data_transformation/Processed_test_data.csv')
columns = test.columns
dtypes = test.dtypes

st.set_page_config(page_title="Risk Profile Prediction App", layout="wide")
st.title('🚗 Auto Insurance Risk Profile Prediction App')

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
    Please enter the following details to predict the risk profile:
""")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['M', 'F'])
    agecat = st.sidebar.selectbox('Age Category', ['1', '2', '3', '4', '5', '6'])
    credit_score = st.sidebar.slider('Credit Score', 300, 850, 500)
    area = st.sidebar.selectbox('Area', ['A', 'B', 'C', 'D'])
    traffic_index = st.sidebar.slider('Traffic Index', 0.0, 5.0, 1.9)
    veh_age = st.sidebar.selectbox('Vehicle Age', ['1', '2', '3', '4', '5'])
    veh_body = st.sidebar.selectbox('Vehicle Body Type', ['SEDAN', 'SUV', 'TRUCK', 'COUPE', 'HATCHBACK'])
    veh_value = st.sidebar.slider('Vehicle Value', 0.0, 5.0, 1.5)
    
    data = {
        'gender': gender,
        'agecat': agecat,
        'credit_score': credit_score,
        'area': area,
        'traffic_index': traffic_index,
        'veh_age': veh_age,
        'veh_body': veh_body,
        'veh_value': veh_value
    }
    return data

user_data = user_input_features()

st.subheader('User Input Features')
st.write("### User Data Summary")
st.json(user_data)

if st.button('Predict Risk Profile'):
    claim_likelihood, claim_amount = risk_profile_model.predict(user_data, columns, dtypes)
    normalized_claim_likelihood, normalized_claim_amount = risk_profile_model.normalize_predictions(claim_likelihood, claim_amount)
    risk_profile = risk_profile_model.classify_risk(normalized_claim_likelihood, normalized_claim_amount, claim_amount)
    
    st.success('Risk Profile Predicted Successfully!')
    
    st.write('### Risk Profile Report')
    st.json(risk_profile)

    st.write("### Detailed Risk Profile")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Claim Likelihood", value=f"{claim_likelihood:.2f}")
        st.metric(label="Normalized Claim Likelihood", value=f"{normalized_claim_likelihood:.2f}")
    
    with col2:
        st.metric(label="Claim Amount", value=f"${claim_amount:.2f}")
        st.metric(label="Normalized Claim Amount", value=f"{normalized_claim_amount:.2f}")
    
    with col3:
        st.metric(label="Risk Profile Probability", value=risk_profile['risk_profile_probability'])
        st.metric(label="Risk Profile Cost", value=risk_profile['risk_profile_cost'])

    with col4:
        st.metric(label="Dynamic Combined Risk Score", value=f"{risk_profile['dynamic_combined_risk_score']:.2f}")
        st.metric(label="Risk Group", value=risk_profile['risk_group'])
