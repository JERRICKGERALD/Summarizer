import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score,mean_absolute_error

from src.exception import CustomException
from src.logger import logging


def save_object(file_path,obj):
    
    try:
        logging.info("Entered Saving object")
        
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited Saving object")
        

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models):

    try:
        report= {}

        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test) 

            train_score = r2_score(y_train,y_train_pred)
            test_score = r2_score(y_test,y_test_pred)

            train_score = mean_absolute_error(y_train,y_train_pred)
            test_score = mean_absolute_error(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_score
        
        return report
    
    

    except Exception as e:
        raise CustomException(e,sys)

