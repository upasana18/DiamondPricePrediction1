import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from source.exception import CustomException
from source.logger import logging
from source.utils import save_object
from source.utils import evaluate_model
from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path =os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 

    def initiate_model_training(self,train_array, test_array):
        try:
            logging.info('spliting dependent and independent data')
            x_train,y_train,x_test,y_test=(train_array[:,:-1],
                                           train_array[:,-1],
                                           test_array[:,:-1],
                                           test_array[:,-1])
            model={'LinearRegression':LinearRegression(),
                   'Lasso':Lasso(),
                   'Ridge':Ridge(),
                   'ElasticNet': ElasticNet(),
                   'DecisionTree':DecisionTreeRegressor(),
                   'RandomForest':RandomForestRegressor()}
            
            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,model)
            print('\n================================================================')
            logging.info(f'model Report:{model_report}')

            best_model_score=max(sorted(model_report.values()))

            best_model_name =list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=model[best_model_name]

            print(f'Best Model Found,model name:{best_model_name},R2 score:{best_model_score}')
            print('\n===================================================================================')
            logging.info(f'Best Model Found,model name:{best_model_name},R2 score:{best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )



        except Exception as e:
            raise CustomException(e,sys)
