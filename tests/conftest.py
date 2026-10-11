from contextlib import ExitStack
from importlib import import_module
from unittest.mock import patch

import pytest
import pandas as pd
import simpliml.timeseriesforecast as tsf


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
    data_modeling_module = import_module('simpliml.timeseriesforecast.dataModeling')
    with patch.object(data_modeling_module.dataModeling, 'buildModel') as mock_build:
        mock_build.return_value = mock_model_result
        yield mock_build


@pytest.fixture
def mock_slow_models():
    stats_module = import_module('simpliml.timeseriesforecast.modelStatsModels')
    pytorch_module = import_module('simpliml.timeseriesforecast.modelPyTorch')
    sklearn_module = import_module('simpliml.timeseriesforecast.modelScikitLearn')
    stats_methods = (
        'modelHoltWinters', 'modelHolt', 'modelSimpleExpSmoothing',
        'modelAutoRegression', 'modelMovingAverage', 'modelARMA',
        'modelARIMA', 'modelSARIMA', 'modelSARIMAX', 'modelVARMA',
        'modelVARMAX',
    )
    mock_result = [['mock', 'mock', True] + [0] * 12]

    with ExitStack() as stack:
        mocks = {
            name: stack.enter_context(patch.object(stats_module.modelStatsModels, name))
            for name in stats_methods
        }
        mocks['lstm'] = stack.enter_context(
            patch.object(pytorch_module.modelPyTorch, 'modelPyTorchLSTM')
        )
        mocks['ensemble'] = stack.enter_context(
            patch.object(sklearn_module.modelScikitLearn, 'runmodelEnsemble')
        )
        for mock in mocks.values():
            mock.return_value = mock_result
        yield mocks


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: mark test as integration test")
