import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scikit_posthocs as sp
import io


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

st.sidebar.header('Select Action')
action = st.sidebar.radio("Choose an action", ["Risk Profile Prediction Report", "Risk Group Analysis Dashboard", "Statistical Analysis of Risk Profiles"])

if action == "Risk Profile Prediction Report":
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

elif action == "Risk Group Analysis Dashboard":
    st.title('📊 Risk Group Analysis Dashboard')

    # Sidebar selection for statistical analysis
    analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Overview", "Detailed Analysis"])

    if analysis_type == "Overview":
        st.write("## Overview of Risk Groups")

        # Calculate and display summary statistics for each risk group
        risk_groups = risk_profile_model.risk_profiles_df['risk_group'].unique()
        for group in risk_groups:
            st.write(f"### {group}")
            group_data = risk_profile_model.risk_profiles_df[risk_profile_model.risk_profiles_df['risk_group'] == group]
            st.write(group_data.describe())

        # Plot distributions
        st.write("### Distributions of Key Features")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        sns.histplot(data=risk_profile_model.risk_profiles_df, x='claim_probability', hue='risk_group', kde=True, ax=axes[0, 0])
        sns.histplot(data=risk_profile_model.risk_profiles_df, x='claim_amount', hue='risk_group', kde=True, ax=axes[0, 1])
        sns.histplot(data=risk_profile_model.risk_profiles_df, x='dynamic_combined_risk_score', hue='risk_group', kde=True, ax=axes[1, 0])
        sns.histplot(data=risk_profile_model.risk_profiles_df, x='credit_score', hue='risk_group', kde=True, ax=axes[1, 1])
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_type == "Detailed Analysis":
        st.write("## Detailed Analysis of Risk Groups")

        # Select a risk group for detailed analysis
        selected_group = st.selectbox("Select Risk Group", risk_profile_model.risk_profiles_df['risk_group'].unique())

        st.write(f"### Detailed Analysis for {selected_group}")
        group_data = risk_profile_model.risk_profiles_df[risk_profile_model.risk_profiles_df['risk_group'] == selected_group]
        st.write(group_data.describe())

        # Plot detailed distributions
        st.write("### Distributions of Key Features")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        sns.histplot(group_data, x='claim_probability', kde=True, ax=axes[0, 0])
        sns.histplot(group_data, x='claim_amount', kde=True, ax=axes[0, 1])
        sns.histplot(group_data, x='dynamic_combined_risk_score', kde=True, ax=axes[1, 0])
        sns.histplot(group_data, x='credit_score', kde=True, ax=axes[1, 1])
        plt.tight_layout()
        st.pyplot(fig)

elif action == "Statistical Analysis of Risk Profiles":
    st.sidebar.title('Statistical Analysis of Risk Profiles')
    analysis_type = st.sidebar.radio("Select Analysis Type", ["Demographic Analysis", "Financial and Regional Analysis", "Vehicular Analysis"])

    def analyze_features(df, features):
        result = {}
        for feature in features:
            if pd.api.types.is_numeric_dtype(df[feature]):
                result[feature] = df[feature].describe().to_frame().reset_index()
            else:
                result[feature] = df[feature].value_counts().to_frame().reset_index()
                result[feature].columns = [feature, 'count']
        return result

    def visualize_features(df, features):
        palette = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
        for feature in features:
            plt.figure(figsize=(12, 6))
            if pd.api.types.is_numeric_dtype(df[feature]):
                fig, ax = plt.subplots()
                sns.boxplot(x='risk_group', y=feature, data=df, palette=palette, ax=ax)
                ax.set_title(f'{feature.capitalize()} by Risk Group')
                ax.set_xlabel('Risk Group')
                ax.set_ylabel(feature.capitalize())
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                sns.countplot(x=feature, hue='risk_group', data=df, palette=palette, ax=ax)
                ax.set_title(f'Risk Group Distribution by {feature.capitalize()}')
                ax.set_xlabel(feature.capitalize())
                ax.set_ylabel('Count')
                st.pyplot(fig)

    def statistical_analysis(df, features):
        results = {}
        analysis_log = io.StringIO()

        for feature in features[:-1]:
            feature_result = {}
            print(f'\nAnalyzing {feature}...', file=analysis_log)
            if pd.api.types.is_numeric_dtype(df[feature]):
                # Normality Test
                grouped = df.groupby('risk_group')
                normality_p_values = {}
                for name, group in grouped:
                    stat, p = stats.shapiro(group[feature])
                    normality_p_values[name] = p
                    print(f'Normality test for {name} - Statistics={stat:.3f}, p-value={p:.3f}', file=analysis_log)
                    feature_result[f'normality_{name}'] = p
                    
                # Homogeneity of variances Test
                groups = [group[feature].dropna() for name, group in grouped]
                stat, p = stats.levene(*groups)
                print(f'Levene’s test for equal variances - Statistics={stat:.3f}, p-value={p:.3f}', file=analysis_log)
                feature_result['levene_p_value'] = p
                
                if all(p_val > 0.05 for p_val in normality_p_values.values()) and p > 0.05:
                    print("Assumptions for ANOVA met, proceeding with ANOVA test.", file=analysis_log)
                    # Perform ANOVA
                    anova_stat, anova_p = stats.f_oneway(*groups)
                    print(f'ANOVA test result: F-Statistic={anova_stat:.3f}, P-Value={anova_p:.3f}', file=analysis_log)
                    feature_result['anova_p_value'] = anova_p
                    
                    if anova_p < 0.05:
                        print('ANOVA significant, performing Tukey HSD test...', file=analysis_log)
                        posthoc = sp.posthoc_tukey_hsd(df[feature], df['risk_group'])
                        print(posthoc, file=analysis_log)
                        feature_result['tukey_hsd'] = posthoc
                    else:
                        print('ANOVA not significant, no further tests required.', file=analysis_log)
                else:
                    print("Assumptions for ANOVA not met, proceeding with Kruskal-Wallis test.", file=analysis_log)
                    # Perform Kruskal-Wallis Test
                    k_stat, k_p = stats.kruskal(*groups)
                    print(f'Kruskal-Wallis Test result: H-Statistic={k_stat:.3f}, P-Value={k_p:.3f}', file=analysis_log)
                    feature_result['kruskal_wallis_p_value'] = k_p
                    
                    if k_p < 0.05:
                        print('Kruskal-Wallis significant, performing Dunn’s test...', file=analysis_log)
                        posthoc = sp.posthoc_dunn(df, val_col=feature, group_col='risk_group', p_adjust='bonferroni')
                        print(posthoc, file=analysis_log)
                        feature_result['dunn_test'] = posthoc
                    else:
                        print('Kruskal-Wallis not significant, no further tests required.', file=analysis_log)
            
            else:
                # Chi-Square Test
                contingency_table = pd.crosstab(df[feature], df['risk_group'])
                chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                print(f"Chi-Square Statistic: {chi2:.2f}", file=analysis_log)
                print(f"Degrees of Freedom: {dof}", file=analysis_log)
                print(f"P-value: {p:.3f}", file=analysis_log)
                print("Expected Frequencies:\n", pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns), file=analysis_log)
                feature_result['chi2_p_value'] = p

                if p < 0.05:
                    print("Chi-Square test significant, calculating standardized residuals...", file=analysis_log)
                    residuals = (contingency_table - expected) / np.sqrt(expected)
                    print("\nStandardized Residuals:\n", residuals, file=analysis_log)
                    feature_result['standardized_residuals'] = residuals
                else:
                    print("Chi-Square test not significant, no further tests required.", file=analysis_log)
            
            results[feature] = feature_result

        analysis_log.seek(0)
        return results, analysis_log.read()

    def display_descriptive_results(results):
         for feature, result in results.items():
            st.write(f"### {feature.capitalize()}")
            st.table(result)   

    def display_hypothesis_analysis_results(results):
        for feature, result in results.items():
            st.write(f"### {feature.capitalize()}")
            for key, value in result.items():
                if isinstance(value, pd.DataFrame):
                    st.write(f"#### {key.capitalize().replace('_', ' ')}")
                    st.table(value)
                else:
                    st.write(f"{key.capitalize().replace('_', ' ')}: {value}")

    if analysis_type == "Demographic Analysis":
        st.write("## Demographic Analysis")
        demographic_features = ['gender', 'agecat', 'risk_group']
        df = risk_profile_model.risk_profiles_df[demographic_features].copy()
        df['agecat'] = df['agecat'].astype(str)
        st.write("### Descriptive Statistics")
        descriptive_stats = analyze_features(df, demographic_features)
        display_descriptive_results(descriptive_stats)
        st.write("### Visualizations")
        visualize_features(df, demographic_features)
        st.write("### Hypothesis Testing")
        hypothesis_testing_results, analysis_log = statistical_analysis(df, demographic_features)
        st.write("#### Analysis Log")
        st.text(analysis_log)
        display_hypothesis_analysis_results(hypothesis_testing_results)

    elif analysis_type == "Financial and Regional Analysis":
        st.write("## Financial and Regional Analysis")
        financial_and_regional_features = ['credit_score', 'area', 'traffic_index', 'risk_group']
        df = risk_profile_model.risk_profiles_df[financial_and_regional_features].copy()
        st.write("### Descriptive Statistics")
        descriptive_stats = analyze_features(df, financial_and_regional_features)
        display_descriptive_results(descriptive_stats)
        st.write("### Visualizations")
        visualize_features(df, financial_and_regional_features)
        st.write("### Hypothesis Testing")
        hypothesis_testing_results, analysis_log = statistical_analysis(df, financial_and_regional_features)
        st.write("#### Analysis Log")
        st.text(analysis_log)
        display_hypothesis_analysis_results(hypothesis_testing_results)

    elif analysis_type == "Vehicular Analysis":
        st.write("## Vehicular Analysis")
        vehicular_features = ['veh_age','veh_body','veh_value', 'risk_group']
        df = risk_profile_model.risk_profiles_df[vehicular_features].copy()
        df['veh_age'] = df['veh_age'].astype(str)
        st.write("### Descriptive Statistics")
        descriptive_stats = analyze_features(df, vehicular_features)
        display_descriptive_results(descriptive_stats)
        st.write("### Visualizations")
        visualize_features(df, vehicular_features)
        st.write("### Hypothesis Testing")
        hypothesis_testing_results, analysis_log = statistical_analysis(df, vehicular_features)
        st.write("#### Analysis Log")
        st.text(analysis_log)
        display_hypothesis_analysis_results(hypothesis_testing_results)
