import pandas as pd
import numpy as np
import os
import logging
import sys
import inspect
import itertools
import datetime
import time
from concurrent import futures
from .tsfCommon import *
from .modelStatsModels import *
from .modelScikitLearn import *
from .modelPyTorch import *


class dataModeling:
    def submitModel(self, modelName, seasonal, modelApproach, x_train, x_test, y_train, y_test, furDF, logVar=''):
        try:
            logging.getLogger(logVar).info("Model Running :: {0}".format(modelName))
            start_time = time.time()
            model_result = eval(modelName + "(seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach)")
            elapsed_time = time.time() - start_time
            return [model_result[0] + [elapsed_time]]
        except Exception as e:
            logging.getLogger(logVar).error("Model Error :: {0}: {1}".format(modelName, str(e)))
            return [[]]
        finally:
            logging.getLogger(logVar).info("Model Completed :: {0}".format(modelName))

    def buildModel(self, dataDF, furDF, seasonal, modelApproach, testSize, runType, logVar=''):
        try:
            columnsName = ['Library', 'Model', 'Status', 'MAE', 'MSE', 'RMSE', 'RMSLE', 'MAPE', 'R2', 'SMAPE', 'MFE',
                           'NMSE', 'THEIL_U_STATISTIC', 'Test', 'Predict', 'TT']
            
            # Fill NaN values once
            dataDF['Data'] = dataDF['Data'].fillna(dataDF['Data'].mean())
            dataDF = dataDF.set_index('Date')
            furDF = furDF.set_index('Date')
            
            # Get train/test split
            train, test = tsfCommon().getTrainTest(dataDF, testSize)
            x_train = train.drop('Data', axis=1)
            x_test = test.drop('Data', axis=1)
            y_train = train['Data']
            y_test = test['Data']

            # Build model list efficiently
            dataVal = []
            mList = []
            
            # Add statsmodels
            for model_name in modelStatsModels().__all__:
                mList.append("modelStatsModels()." + model_name)
            
            # Add scikit-learn
            for model_name in modelScikitLearn("Name").__all__:
                mList.append("modelScikitLearn('" + model_name + "').runmodelEnsemble")
            
            # Add PyTorch models
            for model_name in modelPyTorch().__all__:
                mList.append("modelPyTorch()." + model_name)

            # Remove models with seasonal > 60
            sesonalRmGt60 = ['modelStatsModels().modelSARIMA', 'modelStatsModels().modelSARIMAX']
            if seasonal > 60:
                mList = [m for m in mList if m not in sesonalRmGt60]

            start_time = datetime.datetime.now()
            
            # Prepare repeated parameters efficiently
            repeated_params = {
                'seasonal': itertools.repeat(seasonal),
                'modelApproach': itertools.repeat(modelApproach),
                'x_train': itertools.repeat(x_train),
                'x_test': itertools.repeat(x_test),
                'y_train': itertools.repeat(y_train),
                'y_test': itertools.repeat(y_test),
                'furDF': itertools.repeat(furDF.drop('Data', axis=1))
            }
            
            # Execute models based on run type
            if runType.lower() == "process":
                logging.getLogger(logVar).info("Process Count :: {0}".format(processCount))
                with futures.ProcessPoolExecutor(max_workers=processCount) as executor:
                    results = executor.map(
                        self.submitModel,
                        mList,
                        repeated_params['seasonal'],
                        repeated_params['modelApproach'],
                        repeated_params['x_train'],
                        repeated_params['x_test'],
                        repeated_params['y_train'],
                        repeated_params['y_test'],
                        repeated_params['furDF'],
                        itertools.repeat(logVar)
                    )
                    for result in results:
                        if result and result[0]:
                            dataVal.extend(result)
            else:
                max_workers = threadCount if runType.lower() == "thread" else 1
                logging.getLogger(logVar).info("Thread Count :: {0}".format(max_workers))
                with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = executor.map(
                        self.submitModel,
                        mList,
                        repeated_params['seasonal'],
                        repeated_params['modelApproach'],
                        repeated_params['x_train'],
                        repeated_params['x_test'],
                        repeated_params['y_train'],
                        repeated_params['y_test'],
                        repeated_params['furDF'],
                        itertools.repeat(logVar)
                    )
                    for result in results:
                        if result and result[0]:
                            dataVal.extend(result)

            end_time = datetime.datetime.now()
            logging.getLogger(logVar).info("Total Model Runtime {0} - {1} [ {2} ] ".format(
                start_time, end_time, (end_time - start_time)))

            return pd.DataFrame(dataVal, columns=columnsName)

        except Exception as e:
            logging.getLogger(logVar).error("Error in buildModel: {0}".format(str(e)))
            print(e)
