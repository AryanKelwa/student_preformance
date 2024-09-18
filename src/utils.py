import os
import sys

import pandas as pd
import numpy as np
import pickle

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,X_test,y_train,y_test,models,params):
    try:
        report = {}
        
        for i in range( len(models) ):
            model = list( models.values() )[i]
            model_name = list( models.keys() )[i]
            param = params[model_name]
            gs = GridSearchCV(model,param_grid=param,cv = 3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score( y_train,y_train_pred)
            test_model_score = r2_score( y_test,y_test_pred)
            report[model_name] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
            
            
            
    