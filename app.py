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
from AutoInsurance.pipeline.stage_08_user_app import UserAppPipeline
from AutoInsurance.utils.common import save_json
import os
from fpdf import FPDF
import unicodedata
from AutoInsurance.components.chat_feature import DocumentAnalysis, RiskProfileAnalysis, InsuranceRiskProfileAnalyzer
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


user_app = UserAppPipeline()
risk_profile_model, risk_profiles_df, test = user_app.main()

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

    # st.subheader('User Input Features')
    st.write("### User Data Summary")
    # st.json(user_data)
    user_data_df = pd.DataFrame(list(user_data.items()), columns=['Feature', 'Value']).astype(str)
    st.table(user_data_df)

    if st.button('Predict Risk Profile'):
        claim_likelihood, claim_amount = risk_profile_model.predict(user_data, columns, dtypes)
        normalized_claim_likelihood, normalized_claim_amount = risk_profile_model.normalize_predictions(claim_likelihood, claim_amount)
        risk_profile = risk_profile_model.classify_risk(normalized_claim_likelihood, normalized_claim_amount, claim_likelihood, claim_amount)

        save_json(Path("artifacts/user_app/risk_profile.json"), risk_profile)
        
        st.success('Risk Profile Predicted Successfully!')
        
        #st.write('### Risk Profile Report')
        # st.json(risk_profile)
        risk_profile_df = pd.DataFrame(list(risk_profile.items()), columns=['Feature', 'Value']).astype(str)
        # st.table(risk_profile_df)


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


        ########################### CHAT ROWS #########################

        st.write("Generating description..")
        analyzer = InsuranceRiskProfileAnalyzer()
        context = risk_profile
        response = analyzer.analyze_risk_profile(context)

        st.write(response)


        ########################### END ################################

        




elif action == "Risk Group Analysis Dashboard":
    st.title('📊 Risk Group Analysis Dashboard')

    # Sidebar selection for statistical analysis
    analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Overview", "Detailed Analysis"])


    if analysis_type == "Overview":
        st.write("## Overview of Risk Groups")

        # Calculate and display summary statistics for each risk group
        risk_groups = risk_profiles_df['risk_group'].unique()
        for group in risk_groups:
            st.write(f"### {group}")
            group_data = risk_profiles_df[risk_profiles_df['risk_group'] == group]
            st.write(group_data.iloc[:,1:].describe())

        # Plot distributions
        st.write("### Distributions of Key Features")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        sns.histplot(data=risk_profiles_df, x='claim_probability', hue='risk_group', kde=True, ax=axes[0, 0])
        sns.histplot(data=risk_profiles_df, x='claim_amount', hue='risk_group', kde=True, ax=axes[0, 1])
        sns.histplot(data=risk_profiles_df, x='dynamic_combined_risk_score', hue='risk_group', kde=True, ax=axes[1, 0])
        sns.histplot(data=risk_profiles_df, x='credit_score', hue='risk_group', kde=True, ax=axes[1, 1])
        plt.tight_layout()
        st.pyplot(fig)

    elif analysis_type == "Detailed Analysis":
        st.write("## Detailed Analysis of Risk Groups")

        # Select a risk group for detailed analysis
        selected_group = st.selectbox("Select Risk Group", risk_profiles_df['risk_group'].unique())

        st.write(f"### Detailed Analysis for {selected_group}")
        group_data = risk_profiles_df[risk_profiles_df['risk_group'] == selected_group]
        st.write(group_data.iloc[:,1:].describe())

        # save the risk group to json file
        data_dict = group_data.to_dict(orient='dict')
        filename = f"{selected_group} Profiles.json"
        save_json(Path(f"artifacts/user_app/{filename}"), data_dict)

        # save the risk group to csv file
        group_data.to_csv(Path(f"artifacts/user_app/{selected_group} Profiles.csv"), index=False)

        # Plot detailed distributions
        st.write("### Distributions of Key Features")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        sns.histplot(group_data, x='claim_probability', kde=True, ax=axes[0, 0])
        sns.histplot(group_data, x='claim_amount', kde=True, ax=axes[0, 1])
        sns.histplot(group_data, x='dynamic_combined_risk_score', kde=True, ax=axes[1, 0])
        sns.histplot(group_data, x='credit_score', kde=True, ax=axes[1, 1])
        plt.tight_layout()
        st.pyplot(fig)


        ################################ CHAT #################################

                # Implementing a chat feature in the sidebar
    
        with st.sidebar:
            st.write("## Ask Questions")
            user_query = st.text_input("Enter your question:")
            if user_query:
                analysis = RiskProfileAnalysis(Path("artifacts/user_app/High Risk Profiles.csv"), Path("artifacts/user_app/Low Risk Profiles.csv"), Path("artifacts/user_app/Medium Risk Profiles.csv"))
                data = analysis.load_data()
                db = analysis.setup_database(data['df_high'], data['df_low'], data['df_medium'])
                agent_executor = analysis.setup_chat_agent(db)
                response = analysis.ask_question(agent_executor, user_query)
                st.write(response)





        ################################ END #################################




elif action == "Statistical Analysis of Risk Profiles":
    st.sidebar.title('Statistical Analysis of Risk Profiles')
    analysis_type = st.sidebar.radio("Select Analysis Type", ["Demographic Analysis", "Financial and Regional Analysis", "Vehicular Analysis"])


##################     Start             ############################
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Risk Analysis Report', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def safe_encode(text):
        """ Encode the text to Latin-1, replacing unsupported characters """
        return unicodedata.normalize('NFKD', text).encode('latin-1', 'replace').decode('latin-1')

######################   END             ###################################
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


        figures = []

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



                figures.append(fig)


            else:
                fig, ax = plt.subplots()
                sns.countplot(x=feature, hue='risk_group', data=df, palette=palette, ax=ax)
                ax.set_title(f'Risk Group Distribution by {feature.capitalize()}')
                ax.set_xlabel(feature.capitalize())
                ax.set_ylabel('Count')
                st.pyplot(fig)


                figures.append(fig)



        return figures

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

#####################     start_analysis #################


    def save_analysis_to_pdf(descriptive_stats, visualizations, hypothesis_testing_results, analysis_log, filename):
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add descriptive statistics
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, safe_encode('Descriptive Statistics'), 0, 1, 'L')
        for feature, stat in descriptive_stats.items():
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, safe_encode(feature.capitalize()), 0, 1, 'L')
            stat_string = safe_encode(stat.to_string(index=False))
            pdf.set_font('Arial', size=10)
            pdf.multi_cell(0, 10, stat_string)

        # Save plots with unique filenames
        for i, fig in enumerate(visualizations):
            temp_img_path = f"/tmp/fig_{i}.png"
            fig.savefig(temp_img_path, bbox_inches='tight')
            pdf.add_page()
            pdf.image(temp_img_path, x=None, y=None, w=190, h=100)
            os.remove(temp_img_path)  # Clean up after adding to PDF

        # Hypothesis Testing Results
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, safe_encode('Hypothesis Testing Results'), 0, 1, 'L')
        for feature, results in hypothesis_testing_results.items():
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, safe_encode(feature.capitalize()), 0, 1, 'L')
            for test, result in results.items():
                result_string = safe_encode(f"{test.replace('_', ' ').capitalize()}: {result}")
                pdf.multi_cell(0, 10, result_string)

        # Analysis Log
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, safe_encode('Analysis Log'), 0, 1, 'L')
        pdf.set_font('Arial', size=10)
        pdf.multi_cell(0, 10, safe_encode(analysis_log))

        pdf.output(filename)
############################   End of Analysis Log #########################

    if analysis_type == "Demographic Analysis":
        st.write("## Demographic Analysis")
        demographic_features = ['gender', 'agecat', 'risk_group']
        df = risk_profiles_df[demographic_features].copy()
        df['agecat'] = df['agecat'].astype(str)
        st.write("### Descriptive Statistics")
        descriptive_stats = analyze_features(df, demographic_features)
        display_descriptive_results(descriptive_stats)
        st.write("### Visualizations")
        visualizations = visualize_features(df, demographic_features)
        print(visualizations)
        st.write("### Hypothesis Testing")
        hypothesis_testing_results, analysis_log = statistical_analysis(df, demographic_features)
        st.write("#### Analysis Log")
        st.text(analysis_log)
        display_hypothesis_analysis_results(hypothesis_testing_results)
        filename = Path(f"artifacts/user_app/Demographic Analysis.pdf")
        save_analysis_to_pdf(descriptive_stats, visualizations, hypothesis_testing_results, analysis_log, filename)

        ############################# CHAT ############################


        # Implementing a chat feature in the sidebar
    
        with st.sidebar:
            st.write("## Ask questions")
            user_query = st.text_input("Enter your question:")
            if user_query:
                analysis = DocumentAnalysis(file_path= Path("artifacts/user_app/Demographic Analysis.pdf"))
                pages = analysis.load_document()
                print(f"Document has {len(pages)} pages.")
                splits = analysis.split_text(pages)
                print(f"Document has been split into {len(splits)} text chunks.")
                retriever = analysis.create_vector_store(splits)
                prompt, model = analysis.setup_prompt_and_model()
                question = user_query
                answer = analysis.answer_query(retriever, prompt, model, question)
                st.write(answer)


        ########################### ENF ##############################

    elif analysis_type == "Financial and Regional Analysis":
        st.write("## Financial and Regional Analysis")
        financial_and_regional_features = ['credit_score', 'area', 'traffic_index', 'risk_group']
        df = risk_profiles_df[financial_and_regional_features].copy()
        st.write("### Descriptive Statistics")
        descriptive_stats = analyze_features(df, financial_and_regional_features)
        display_descriptive_results(descriptive_stats)
        st.write("### Visualizations")
        visualizations = visualize_features(df, financial_and_regional_features)
        st.write("### Hypothesis Testing")
        hypothesis_testing_results, analysis_log = statistical_analysis(df, financial_and_regional_features)
        st.write("#### Analysis Log")
        st.text(analysis_log)
        display_hypothesis_analysis_results(hypothesis_testing_results)
        filename = Path(f"artifacts/user_app/Financial and Regional Analysis.pdf")
        save_analysis_to_pdf(descriptive_stats, visualizations, hypothesis_testing_results, analysis_log, filename)

        with st.sidebar:
            st.write("## Ask Questions")
            user_query = st.text_input("Enter your question:")
            if user_query:
                analysis = DocumentAnalysis(file_path= Path("artifacts/user_app/Financial and Regional Analysis.pdf"))
                pages = analysis.load_document()
                print(f"Document has {len(pages)} pages.")
                splits = analysis.split_text(pages)
                print(f"Document has been split into {len(splits)} text chunks.")
                retriever = analysis.create_vector_store(splits)
                prompt, model = analysis.setup_prompt_and_model()
                question = user_query
                answer = analysis.answer_query(retriever, prompt, model, question)
                st.write(answer)


    elif analysis_type == "Vehicular Analysis":
        st.write("## Vehicular Analysis")
        vehicular_features = ['veh_age','veh_body','veh_value', 'risk_group']
        df = risk_profiles_df[vehicular_features].copy()
        df['veh_age'] = df['veh_age'].astype(str)
        st.write("### Descriptive Statistics")
        descriptive_stats = analyze_features(df, vehicular_features)
        display_descriptive_results(descriptive_stats)
        st.write("### Visualizations")
        visualizations = visualize_features(df, vehicular_features)
        st.write("### Hypothesis Testing")
        hypothesis_testing_results, analysis_log = statistical_analysis(df, vehicular_features)
        st.write("#### Analysis Log")
        st.text(analysis_log)
        display_hypothesis_analysis_results(hypothesis_testing_results)
        filename = Path(f"artifacts/user_app/Vehicular Analysis.pdf")
        save_analysis_to_pdf(descriptive_stats, visualizations, hypothesis_testing_results, analysis_log, filename)



        with st.sidebar:
            st.write("## Ask Questions")
            user_query = st.text_input("Enter your question:")
            if user_query:
                analysis = DocumentAnalysis(file_path= Path("artifacts/user_app/Vehicular Analysis.pdf"))
                pages = analysis.load_document()
                print(f"Document has {len(pages)} pages.")
                splits = analysis.split_text(pages)
                print(f"Document has been split into {len(splits)} text chunks.")
                retriever = analysis.create_vector_store(splits)
                prompt, model = analysis.setup_prompt_and_model()
                question = user_query
                answer = analysis.answer_query(retriever, prompt, model, question)
                st.write(answer)
        




##################################################################################################

