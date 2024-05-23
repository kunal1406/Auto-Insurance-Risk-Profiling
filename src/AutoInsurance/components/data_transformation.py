import pandas as pd
import numpy as np
import datetime
from AutoInsurance import logger
from AutoInsurance.entity.config_entity import DataTransformationConfig
import os


REFERENCE_DATE = datetime.datetime(2017, 12, 31)

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def calculate_age(self, dob):
        '''Compute the age of a person as of December 31, 2017.'''
        return REFERENCE_DATE.year - dob.year - ((REFERENCE_DATE.month, REFERENCE_DATE.day) < (dob.month, dob.day))

    def map_age_to_category(self, ages):
        '''Map ages to categories using vectorized operations.'''
        bins = [0, 18, 28, 38, 48, 58, 68, float('inf')]
        labels = range(len(bins) - 1)
        return pd.cut(ages, bins=bins, labels=labels, right=False)

    def impute_with_median(self, df, column, groupby_column):
        '''Impute missing values in a column using the median of groups defined by another column.'''
        medians = df.groupby(groupby_column, observed=True)[column].transform('median')
        df[column] = df[column].fillna(medians)
        return df

    def prepare_and_clean_data(self, df):
        '''Prepare and clean the DataFrame.'''
        df["date_of_birth"] = pd.to_datetime(df['date_of_birth'])
        df['age'] = df['date_of_birth'].apply(self.calculate_age)
        if 'agecat' in df.columns and df['agecat'].isnull().any():
            df['agecat'] = self.map_age_to_category(df['age'])
        df = self.impute_with_median(df, 'credit_score', 'agecat')
        df = self.impute_with_median(df, 'traffic_index', 'area')
        df['veh_value'] = np.log(df['veh_value'] + 1)
        df['agecat'] = df['agecat'].astype('object')
        df['veh_age'] = df['veh_age'].astype('object')
        return df

    def get_dummies(self, df):
        '''Get dummy variables for categorical features.'''
        return pd.get_dummies(df, columns=['gender', 'area', 'veh_body', 'agecat', 'veh_age'], drop_first=True)

    def transform_for_classification(self, df_2017, df_2018):
        '''Transform data for classification.'''
        df_2017 = self.get_dummies(df_2017)
        df_2018 = self.get_dummies(df_2018)

        df_2017['claim'] = df_2017['numclaims'].apply(lambda x: 0 if x == 0 else 1)

        X = df_2017.drop(["numclaims", "claimcst0", "claim"], axis=1)
        y = df_2017["claim"]
        x_test = df_2018

        return X, y, x_test

    def transform_for_regression(self, df_2017, df_2018):
        '''Transform data for regression.'''
        df_2017 = self.get_dummies(df_2017)
        df_2018 = self.get_dummies(df_2018)

        df_2017['claim'] = df_2017['numclaims'].apply(lambda x: 0 if x == 0 else 1)

        claim_amount_train = df_2017[df_2017['claim'] > 0].copy()
        claim_amount_train['amountperclaim'] = np.where(
            claim_amount_train['numclaims'] > 0, 
            claim_amount_train['claimcst0'] / claim_amount_train['numclaims'],
            0  
        )

        claim_amount_train["log_amount"]=(claim_amount_train.amountperclaim+1).apply(np.log)

        X_reg = claim_amount_train.drop(["numclaims", "claimcst0", "claim", "amountperclaim", "log_amount"], axis=1)
        y_reg = claim_amount_train["log_amount"]

        x_test = df_2018

        return X_reg, y_reg, x_test

    def load_and_transform_data(self):
        '''Load data and apply transformations.'''
        df_2017 = pd.read_csv(self.config.train_data_path, parse_dates=True)
        df_2018 = pd.read_csv(self.config.test_data_path, parse_dates=True)
        
        clean_data_2017 = self.prepare_and_clean_data(df_2017)
        clean_data_2018 = self.prepare_and_clean_data(df_2018)
        
        data_2017 = clean_data_2017.drop(["age", "claim_office", "pol_number", "pol_eff_dt", "annual_premium", "date_of_birth"], axis=1)
        data_2018 = clean_data_2018.drop(["quote_number", "date_of_birth", "age"], axis=1)
        
        X_class, y_class, x_test_class = self.transform_for_classification(data_2017, data_2018)
        X_reg, y_reg, x_test_reg = self.transform_for_regression(data_2017, data_2018)

        logger.info("Transformed the data as per required by the models respectively.")

        class_train_data = X_class.copy()
        class_train_data['claim'] = y_class
        
        reg_train_data = X_reg.copy()
        reg_train_data['log_amount'] = y_reg

        # Save the processed data
        clean_data_2018.to_csv(os.path.join(self.config.root_dir, "potential_customers.csv"), index=False)
        class_train_data.to_csv(os.path.join(self.config.root_dir,"processed_train_class_data.csv"), index=False)
        reg_train_data.to_csv(os.path.join(self.config.root_dir,"processed_train_reg_data.csv"), index=False)
        x_test_class.to_csv(os.path.join(self.config.root_dir,"Processed_test_data.csv"), index=False)
        
        logger.info("Processed files saved to their respective paths")
        
        return (X_class, y_class, x_test_class), (X_reg, y_reg, x_test_reg)