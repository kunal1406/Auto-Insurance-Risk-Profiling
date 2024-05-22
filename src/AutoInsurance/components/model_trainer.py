import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
import seaborn as sns
import math
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, mean_squared_error, r2_score, classification_report
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import scipy.stats as stats
import joblib
import os
from AutoInsurance.entity.config_entity import (ClassModelTrainerConfig, RegModelTrainerConfig)


class ClassModelTrainer:
    def __init__(self, config: ClassModelTrainerConfig):
        self.config = config

    def train_model(self):
        train_data = pd.read_csv(self.config.train_data_class_path)
        test_data = pd.read_csv(self.config.test_data_path)
        train_x = train_data.drop('claim', axis = 1)
        train_y = train_data['claim']

        model = GradientBoostingClassifier(
            n_estimators= self.config.n_estimators,
            learning_rate= self.config.learning_rate,
            max_depth= self.config.max_depth,
            min_samples_leaf= self.config.min_samples_leaf,
            max_features= self.config.max_features,
        )

        model.fit(train_x, train_y)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_class_name))


class RegModelTrainer:
    def __init__(self, config: RegModelTrainerConfig):
        self.config = config

    def train_model(self):
        train_data = pd.read_csv(self.config.train_data_reg_path)
        test_data = pd.read_csv(self.config.test_data_path)
        train_x = train_data.drop('log_amount', axis = 1)
        train_y = train_data['log_amount']

        model = GradientBoostingRegressor(
            n_estimators= self.config.n_estimators,
            learning_rate= self.config.learning_rate,
            max_depth= self.config.max_depth,
            min_samples_leaf= self.config.min_samples_leaf,
            max_features= self.config.max_features,
        )

        model.fit(train_x, train_y)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_reg_name))
