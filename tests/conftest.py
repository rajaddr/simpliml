import pytest
import pandas as pd
import numpy as np
import simpliml.timeseriesforecast as tsf
from unittest.mock import patch


@pytest.fixture(scope="session")
def session_air_data():
    # Small sample AirPassengers-like data
    return pd.DataFrame({'Date': ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05'],
                         'Data': [100, 110, 105, 115, 120]})


@pytest.fixture(scope="session")
def session_model_data(session_air_data):
    dataDF, futureDF = tsf.generateTSData(session_air_data, format='%Y-%m', freq='MS', periods=6)
    return dataDF, futureDF


@pytest.fixture(scope="session")
def session_ts_data(session_air_data):
    dataDF, _ = tsf.generateTSData(session_air_data, format='%Y-%m', freq='MS', periods=6)
    return dataDF


@pytest.fixture
def valid_data(session_air_data):
    return session_air_data.copy()


@pytest.fixture
def model_data(session_model_data):
    return (session_model_data[0].copy(), session_model_data[1].copy())


@pytest.fixture
def valid_ts_data(session_ts_data):
    return session_ts_data.copy()


@pytest.fixture
def mock_model_result():
    return pd.DataFrame({
        'Model': ['ARIMA'],
        'Status': [True],
        'MAPE': [0.05],
        'MAE': [1.0],
        'MSE': [1.5],
        'RMSE': [1.2],
        'RMSLE': [0.1],
        'R2': [0.95],
        'SMAPE': [0.04],
        'MFE': [0.1],
        'NMSE': [0.01],
        'THEIL_U_STATISTIC': [0.02],
        'Test': [[['2021-01', 100]]],
        'Predict': [[['2021-02', 105]]]
    })


@pytest.fixture
def mock_fast_model(mock_model_result):
    with patch('simpliml.timeseriesforecast.dataModeling.dataModeling.buildModel') as mock_build:
        mock_build.return_value = mock_model_result
        yield mock_build


@pytest.fixture
def mock_slow_models():
    with patch('simpliml.timeseriesforecast.modelStatsModels.modelStatsModels.modelARIMA', create=True) as mock_arima, \
         patch('simpliml.timeseriesforecast.modelStatsModels.modelStatsModels.modelSARIMA', create=True) as mock_sarima, \
         patch('simpliml.timeseriesforecast.modelStatsModels.modelStatsModels.modelAutoARIMA', create=True) as mock_auto, \
         patch('simpliml.timeseriesforecast.modelStatsModels.modelStatsModels.modelExponentSmoothing', create=True) as mock_exp, \
         patch('simpliml.timeseriesforecast.modelPyTorch.modelPyTorch.runModelPyTorchLSTM', create=True) as mock_lstm, \
         patch('simpliml.timeseriesforecast.modelScikitLearn.modelScikitLearn.runmodelEnsemble', create=True) as mock_ensemble:

        mock_arima.return_value = [True, 0.05, 100, 105]
        mock_sarima.return_value = [True, 0.04, 100, 104]
        mock_auto.return_value = [True, 0.045, 100, 104.5]
        mock_exp.return_value = [True, 0.06, 100, 106]
        mock_lstm.return_value = np.array([105, 106, 107])
        mock_ensemble.return_value = [True, 0.035, 100, 103.5]

        yield {
            'arima': mock_arima,
            'sarima': mock_sarima,
            'auto': mock_auto,
            'exp': mock_exp,
            'lstm': mock_lstm,
            'ensemble': mock_ensemble
        }


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: mark test as integration test")
