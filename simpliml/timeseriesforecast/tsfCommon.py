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
        """
        Optimized evaluation metrics calculation using vectorized operations.
        """
        roundVal = 3
        
        # Forward fill and backward fill only once
        y = y.bfill().ffill()
        predictions = predictions.bfill().ffill()
        
        # Convert to numpy for vectorized operations
        y_np = np.asarray(y, dtype=float)
        pred_np = np.asarray(predictions, dtype=float)
        
        # Calculate differences once
        diff = y_np - pred_np
        abs_diff = np.abs(diff)
        
        # Vectorized calculations
        mae = np.mean(abs_diff)
        mse = np.mean(diff ** 2)
        rmse = np.sqrt(mse)
        
        # SMAPE: Symmetric Mean Absolute Percentage Error
        denominator = np.abs(y_np) + np.abs(pred_np)
        smape = np.mean(np.where(denominator != 0, 2 * abs_diff / denominator, 0)) * 100
        
        # MAPE: Mean Absolute Percentage Error
        mape = np.mean(np.where(y_np != 0, abs_diff / np.abs(y_np), 0)) * 100
        
        # MFE: Mean Forecast Error
        mfe = np.mean(diff)
        
        # NMSE: Normalized Mean Squared Error
        y_mean = np.mean(y_np)
        nmse = mse / (np.sum((y_np - y_mean) ** 2) / (len(y_np) - 1)) if len(y_np) > 1 else 0
        
        # Theil U Statistic
        rmse_error = np.sqrt(np.mean(diff ** 2))
        pred_rms = np.sqrt(np.mean(pred_np ** 2))
        y_rms = np.sqrt(np.mean(y_np ** 2))
        theil_u_statistic = rmse_error / (pred_rms * y_rms) if (pred_rms * y_rms) != 0 else 0
        
        # R² Score
        r2 = r2_score(y_np, pred_np)
        
        # RMSLE: Root Mean Squared Log Error
        rmsle = mean_squared_log_error(y_np, pred_np, multioutput='raw_values')[0]

        return [round(mae, roundVal), round(mse, roundVal), round(rmse, roundVal), round(rmsle, roundVal),
                round(mape, roundVal), round(r2, roundVal),
                round(smape, roundVal), round(mfe, roundVal),
                round(nmse, roundVal), round(theil_u_statistic, roundVal)]

    def getTrainTest(self, dataDF, testSize):
        """Split data into train and test sets efficiently."""
        train_size = int(len(dataDF) * testSize)
        return dataDF[:train_size], dataDF[train_size:]
