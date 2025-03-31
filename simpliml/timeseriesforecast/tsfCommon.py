import pandas as pd, numpy as np, math, time, itertools, inspect, warnings, os
from random import random

from sklearn.metrics import *

from statsmodels.tsa.holtwinters import *
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
#from pmdarima.arima import auto_arima

threadCount = os.cpu_count() * 2
processCount = 1 if os.cpu_count() == 1 else round(os.cpu_count() / 4)


class tsfCommon:
    def modelEval(self, y, predictions):
        roundVal = 3
        y = y.bfill().ffill()
        predictions = predictions.bfill().ffill()
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        SMAPE = np.mean(np.abs((y - predictions) / ((y + predictions) / 2))) * 100
        rmse = np.sqrt(mean_squared_error(y, predictions))
        MAPE = np.mean(np.abs((y - predictions) / y)) * 100
        mfe = np.mean(y - predictions)
        NMSE = mse / (np.sum((y - np.mean(y)) ** 2) / (len(y) - 1))
        error = y - predictions
        mfe = np.sqrt(np.mean(predictions ** 2))
        mse = np.sqrt(np.mean(y ** 2))
        rmse = np.sqrt(np.mean(error ** 2))
        theil_u_statistic = rmse / (mfe * mse)
        r2 = r2_score(y, predictions)
        RMSLE = mean_squared_log_error(y, predictions, multioutput='raw_values')[0]

        return [round(mae, roundVal), round(mse, roundVal), round(rmse, roundVal), round(RMSLE, roundVal),
                round(MAPE, roundVal), round(r2, roundVal),
                round(SMAPE, roundVal), round(mfe, roundVal),
                round(NMSE, roundVal), round(theil_u_statistic, roundVal)]

    def getTrainTest(self, dataDF, testSize):
        train_size = int(len(dataDF) * (testSize))
        return dataDF[0:train_size], dataDF[train_size:]
