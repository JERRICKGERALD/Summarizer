import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        
        logging.info("Model trainer started")

        try:

            logging.info("Splitting training and test input data")

            X_train, y_train,X_test,y_test = (
                train_arr[:,:-1], 
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                
                )

            models = {

                    "RandomForest":RandomForestRegressor(),
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "XGBoost Regressor": XGBRegressor(),
                    "CatBoost Regressor": CatBoostRegressor(verbose=False),
            }
            
            model_report:dict = evaluate_model(X_train = X_train, X_test = X_test, y_train = y_train, y_test= y_test,models = models)

            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square
        
        except Exception as e:
            
            logging.info("Exception occured at Model Training")
            raise CustomException(e,sys)
