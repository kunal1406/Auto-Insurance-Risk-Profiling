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
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from pathlib import Path
from AutoInsurance.utils.common import save_json
from AutoInsurance.entity.config_entity import (ClassModelEvaluationConfig, RegModelEvaluationConfig)
from AutoInsurance.utils.common import logger


class ClassModelEvaluation:
    def __init__(self, config: ClassModelEvaluationConfig):
        self.config = config

    def perform_k_fold(self, X, y):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=45)
        cv_scores = []
        pred_full = np.zeros(y.shape[0]) 
        true_full = np.zeros(y.shape[0]) 

        i = 1

        for train_index, test_index in kf.split(X, y):
            print(f"Fold {i} started of {kf.n_splits}")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            gb = GradientBoostingClassifier(learning_rate=0.1,max_depth=4,max_features=0.3,min_samples_leaf=5,n_estimators=100)
            gb.fit(X_train, y_train)
            pred_probs = gb.predict_proba(X_test)[:, 1]

            pred_full[test_index] = pred_probs  
            true_full[test_index] = y_test  

            score = roc_auc_score(y_test, pred_probs)
            print('roc_auc_score', score)
            cv_scores.append(score)

            i += 1
        
        fpr, tpr, thresholds = roc_curve(true_full, pred_full)
        auc_val = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print("optimal threshold is", optimal_threshold)

        predicted_labels = (pred_full >= optimal_threshold)
        report = classification_report(true_full, predicted_labels, output_dict=True)
        print(report)

        return gb, cv_scores, optimal_threshold, report

    def evaluate_model(self, X, y):
        gb, cv_scores, optimal_threshold, report = self.perform_k_fold(X, y)
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        print(f"Mean roc_auc_score: {mean_score}")
        print(f"Std roc_auc_score: {std_score}")
        return gb, mean_score, std_score, optimal_threshold, report
    def log_into_mlflow(self):

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            data = pd.read_csv(self.config.train_data_path)
            X = data.drop('claim', axis=1)
            print(X.shape)
            y = data['claim']
            print(y.shape)
            model, roc_auc_score, std_roc_auc_score, optimal_threshold, report = self.evaluate_model(X, y)

            scores = {"roc_auc_score": roc_auc_score, "optimal_threshold": optimal_threshold}
            save_json(path=Path(self.config.class_metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("roc_auc_score", roc_auc_score)
            mlflow.log_metric("std roc_auc_score", std_roc_auc_score)
            mlflow.log_metric("optimal_threshold", optimal_threshold)

            for label, metric in report.items():
                if label not in ["accuracy", "macro avg", "weighted avg"]:
                    for metric_name, metric_value in metric.items():
                        mlflow.log_metric(f"{label}_{metric_name}", metric_value)

            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="GradientBoostingClassifier")
            else:
                mlflow.sklearn.log_model(model, "model")


class RegModelEvaluation:
    def __init__(self, config: RegModelEvaluationConfig):
        self.config = config

    def perform_k_fold(self, X, y):
        model = GradientBoostingRegressor(
            learning_rate=0.1,
            max_depth=4,
            max_features=0.3,
            min_samples_leaf=5,
            n_estimators=100
        )
        kf = KFold(n_splits=10, shuffle=True, random_state=45)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

        return model, cv_scores
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        return r2, rmse, mae, predictions
    
    def log_into_mlflow(self):

        try:
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                data = pd.read_csv(self.config.train_data_path)
                X = data.drop('log_amount', axis=1)
                y = data['log_amount']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

                model, cv_scores = self.perform_k_fold(X, y)

                r2, rmse, mae, predictions = self.evaluate_model(model, X_train, y_train, X_test, y_test)

                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                save_json(path=Path(self.config.reg_metric_file_name), data=scores)

                mlflow.log_params(self.config.all_params)
                mlflow.log_metric("mean_cv_r2_score", np.mean(cv_scores))
                mlflow.log_metric("std_cv_r2_score", np.std(cv_scores))
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)

                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="GradientBoostingRegressor")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            logger.exception(f"error logging to MLflow: {e}")
            
