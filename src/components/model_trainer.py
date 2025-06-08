import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data.')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regression": DecisionTreeRegressor(),
                "K-Neighbours Regression": KNeighborsRegressor(),
                "Random Forest Regression": RandomForestRegressor(),
                "AdaBoost Regression": AdaBoostRegressor(),
                "Gradient Regression": GradientBoostingRegressor(),
                "XgBoost Regression": XGBRegressor(),
                "CatBoost Regression": CatBoostRegressor()
            }

            params = {
                "Linear Regression": {}, 
                "Decision Tree Regression": {
                    "criterion": ["squared_error", "absolute_error"]
                },
                "K-Neighbours Regression": {
                    "n_neighbors": [3, 5, 7]
                },
                "Random Forest Regression": {
                    "n_estimators": [50, 100]
                },
                "AdaBoost Regression": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "loss": ["linear", "square"]
                },
                "Gradient Regression": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                },
                "XgBoost Regression": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                },
                "CatBoost Regression": {
                    "iterations": [50, 100],
                    "depth": [4, 6],
                    "learning_rate": [0.01, 0.1]
                }
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models, params = params)

            ## to get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## to get best model name ffrom dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model: {best_model_name} with R² Score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException('No best model found.')
            
            logging.info(f'Best found model on both training and testing dataset.')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return f"Best r2 score of predicted data: {r2_square}"
        

        except Exception as e:
            raise CustomException(e, sys)        

