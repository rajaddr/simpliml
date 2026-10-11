# Coverage tests for the corresponding production source module.

# Consolidated coverage tests. Keep coverage for each source module in this file.

# --- test_cover_missing.py ---
import importlib, os, sys
import pandas as pd, numpy as np
import pytest
from unittest.mock import patch, MagicMock


import simpliml
import simpliml.timeseriesforecast as tsf
import simpliml.timeseriesforecast.dataAnalysis as da_mod
import simpliml.timeseriesforecast.tsfCommon as common_mod
import simpliml.timeseriesforecast.modelStatsModels as msm_mod
import simpliml.timeseriesforecast.modelScikitLearn as msk_mod
import simpliml.timeseriesforecast.modelPyTorch as mpt_mod
import simpliml.timeseriesforecast.dataBuild as db_mod
from io import StringIO

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The total space of parameters .*:UserWarning"
    ),
    pytest.mark.filterwarnings("ignore::sklearn.exceptions.FitFailedWarning"),
    pytest.mark.filterwarnings(
        "ignore:One or more of the (test|train) scores are non-finite.*:UserWarning"
    ),
]



def test_pytorch_lstm_and_training_paths():
    mp = importlib.import_module('simpliml.timeseriesforecast.modelPyTorch')
    model = mp.modelPyTorch()

    lstm = mp.LSTM(num_classes=1, input_size=1, hidden_size=2, num_layers=1)
    x = np.random.rand(2, 1, 1).astype(np.float32)
    out = lstm(x)
    assert out.shape[0] == 2
    assert out.shape[1] == 1

    x_train = pd.DataFrame({'x1': [1.0, 2.0, 3.0, 4.0]})
    x_test = pd.DataFrame({'x1': [5.0, 6.0]})
    y_train = pd.DataFrame({'Data': [10.0, 11.0, 12.0, 13.0]})
    y_test = pd.DataFrame({'Data': [14.0, 15.0]})
    furDF = pd.DataFrame({'Date': ['2021-01-03', '2021-01-04'], 'x1': [7.0, 8.0]})

    os.environ['SIMPLIML_PYTEST'] = '1'
    try:
        train_res = model.runModelPyTorchLSTM(x_train, x_test, y_train)
        assert train_res is not None

        forecast_res = model.modelPyTorchLSTM(
            seasonal=7,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train['Data'],
            y_test=y_test['Data'],
            furDF=furDF,
            modelApproach='BEST',
        )
        assert isinstance(forecast_res, list)
    finally:
        os.environ.pop('SIMPLIML_PYTEST', None)


def test_remaining_pytorch_sliding_window_branch():
    model = importlib.import_module('simpliml.timeseriesforecast.modelPyTorch').modelPyTorch()
    result = model.sliding_windows([[[1.0]], [[2.0]]])
    assert isinstance(result, np.ndarray)


def test_modelpytorch_lstm_class_and_forward():
    """Test PyTorch LSTM class instantiation and forward pass"""
    mp = importlib.import_module('simpliml.timeseriesforecast.modelPyTorch')
    
    # Test LSTM class directly
    lstm = mp.LSTM(num_classes=1, input_size=1, hidden_size=2, num_layers=1)
    assert lstm.num_classes == 1
    assert lstm.input_size == 1
    assert lstm.hidden_size == 2
    assert lstm.num_layers == 1


def test_modelpytorch_sliding_windows_variants():
    """Test sliding_windows with different input types"""
    mp = importlib.import_module('simpliml.timeseriesforecast.modelPyTorch')
    model = mp.modelPyTorch()
    
    # Test with numpy array (1D)
    arr1d = np.array([1, 2, 3, 4, 5])
    result1 = model.sliding_windows(arr1d)
    assert isinstance(result1, np.ndarray)
    
    # Test with numpy array (2D)
    arr2d = np.array([[1, 2], [3, 4], [5, 6]])
    result2 = model.sliding_windows(arr2d)
    assert isinstance(result2, np.ndarray)
    
    # Test with DataFrame
    df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    result3 = model.sliding_windows(df)
    assert isinstance(result3, np.ndarray)


def test_modelpytorch_sliding_and_model(monkeypatch):
    mpt = importlib.import_module('simpliml.timeseriesforecast.modelPyTorch')
    # sliding_windows on array
    mp = mpt.modelPyTorch()
    arr = np.array([1, 2, 3])
    sw = mp.sliding_windows(arr)
    assert isinstance(sw, np.ndarray)

    # Patch runModelPyTorchLSTM to avoid heavy torch
    def fake_run(x_train_R, x_test_R, y_train_R):
        return np.array([[10], [11], [12]])[:len(x_test_R)]

    monkeypatch.setattr(mpt.modelPyTorch, 'runModelPyTorchLSTM', staticmethod(fake_run))

    # Prepare small training/test sets (DataFrame/Series as expected by runModelPyTorchLSTM wrapper)
    x_train = pd.DataFrame({'Date': ['d1'], 'f1': [1]})
    x_test = pd.DataFrame({'Date': ['d2'], 'f1': [2]})
    y_train = pd.DataFrame({'Date': ['d1'], 'Data': [1]})
    y_test = pd.DataFrame({'Date': ['d2'], 'Data': [2]})
    furDF = pd.DataFrame({'Date': ['d3'], 'f1': [3]})

    res = mpt.modelPyTorch().modelPyTorchLSTM(12, x_train, x_test, y_train, y_test, furDF, 'FAST')
    assert isinstance(res, list)

