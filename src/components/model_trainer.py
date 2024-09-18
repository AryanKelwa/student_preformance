import os
import sys

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('splitting the training and test input data')
            
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
            'LinearRegression':LinearRegression(),
            'KNeighborsRegressor':KNeighborsRegressor(),
            'DecisionTreeRegressor':DecisionTreeRegressor(),
            'RandomForestRegressor' : RandomForestRegressor(),
            'XGBRegressor' : XGBRegressor(),
            'AdaBoostRegressor' : AdaBoostRegressor(),
            'GradientBoostingRegressor':GradientBoostingRegressor()
            }
            
            models_report = evaluate_model(X_train=X_train,
                                           X_test = X_test,
                                           y_train = y_train,
                                           y_test = y_test,
                                           models=models)
            
            best_model_score = max( sorted( models_report.values() ) )
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                logging.info('No best model found')
                raise CustomException('No best Model Found')

            logging.info('Best model found')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                
            )
            logging.info('best model object saved')
            
            # Training of the Model
            best_model.fit(X_train,y_train)
            
            #Prediction of the Model
            predicted = best_model.predict(X_test)
            #r-squared error
            r_square = r2_score(y_test,predicted)
            
            return r_square            
            
        except Exception as e:
            raise CustomException(e,sys)