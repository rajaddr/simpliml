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



def test_package_functions_return_expected_result_or_minusone():
    tsf = importlib.import_module('simpliml.timeseriesforecast')
    sample = pd.DataFrame({'Date': ['2021-01-01', '2021-01-02'], 'Data': [10.0, 12.0]})

    assert tsf.generateTSData(sample, freq='D', periods=2)[0] is not None
    assert tsf.analysisData(sample) is None
    assert tsf.modelResult(sample, pd.DataFrame({'Status': [True], 'MAPE': [1.0], 'Model': ['A'], 'Test': [[['2021-01-01', 10.0]]], 'Predict': [[['2021-01-03', 11.0]]]}), 'BEST') is not None

    empty = pd.DataFrame({'Date': [], 'Data': []})
    assert tsf.generateTSData(empty, freq='D', periods=2) == (-1, -1)
    assert tsf.analysisData(empty) == -1
    assert tsf.modelResult(empty, pd.DataFrame()) == -1


def test_about_contents():
    info = simpliml.about.get_about()
    for key in ['__version__', '__author__', '__description__']:
        assert key in info


def test_tsf_init_wrappers(monkeypatch):
    tsf = importlib.import_module('simpliml.timeseriesforecast')
    dm_module = importlib.import_module('simpliml.timeseriesforecast.dataModeling')

    # Patch dataModeling.buildModel to return DataFrame quickly
    def fake_build(self, dataDF, furDF, seasonal, modelApproach, testSize, runType, logVar=''):
        return pd.DataFrame({'Model': ['X'], 'Status': [True]})

    monkeypatch.setattr(dm_module.dataModeling, 'buildModel', fake_build, raising=False)

    # small dataframes
    df = pd.DataFrame({'Date': ['2021-01', '2021-02'], 'Data': [1, 2]})
    fur = pd.DataFrame({'Date': ['2021-03'], 'Data': [3]})

    # exercise wrappers
    assert tsf.buildModel(df, fur, seasonal=7, modelApproach='FAST', testSize=80, runType='Single') is not None
    assert tsf.runModel(df, fur, seasonal=7, modelApproach='FAST', testSize=80) is not None
    assert tsf.runThreadModel(df, fur, seasonal=7, modelApproach='FAST', testSize=80) is not None
    assert tsf.runProcessModel(df, fur, seasonal=7, modelApproach='FAST', testSize=80) is not None


def test_timeseries_init_branches(monkeypatch):
    tsf = importlib.import_module('simpliml.timeseriesforecast')
    da_module = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')
    dm_module = importlib.import_module('simpliml.timeseriesforecast.dataModeling')

    # generateTSData with wrong shape should return -1, -1
    bad = pd.DataFrame({'A': [1, 2, 3]})
    assert tsf.generateTSData(bad) == (-1, -1)

    # analysisData: force underlying analysis to raise to hit except branch
    def raise_analysis(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(da_module.dataAnalysis, 'analysisData', raise_analysis, raising=False)
    df_multi = pd.DataFrame({'Date': ['2021-01','2021-02'], 'Data': [1,2]})
    # wrapper should catch and return -1 because underlying analysis raises
    assert tsf.analysisData(df_multi) == -1

    # runModel/runThreadModel/runProcessModel else branch when small data
    small = pd.DataFrame({'Date': ['2021-01'], 'Data': [1]})
    small_fur = pd.DataFrame({'Date': [], 'Data': []})
    assert tsf.runModel(small, small_fur) == -1
    assert tsf.runThreadModel(small, small_fur) == -1
    assert tsf.runProcessModel(small, small_fur) == -1

    # force dataModeling.buildModel to raise to hit except in wrapper
    def raise_build(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(dm_module.dataModeling, 'buildModel', raise_build, raising=False)
    df2 = pd.DataFrame({'Date': ['2021-01', '2021-02'], 'Data': [1,2]})
    fur2 = pd.DataFrame({'Date': ['2021-03'], 'Data': [3]})
    assert tsf.runModel(df2, fur2) == -1
    assert tsf.runThreadModel(df2, fur2) == -1
    assert tsf.runProcessModel(df2, fur2) == -1

    # modelResult: empty mdlResult -> else branch
    empty = pd.DataFrame({'A': []})
    assert tsf.modelResult(df2, empty) == -1


def test_tsf_module_level_patching(monkeypatch):
    tsf = importlib.import_module('simpliml.timeseriesforecast')

    # monkeypatch the dataAnalysis symbol in tsf module to a class that raises
    class BadDA:
        def analysisData(self, *a, **k):
            raise RuntimeError('boom')
        def modelResult(self, *a, **k):
            raise RuntimeError('boom')

    monkeypatch.setattr(tsf, 'dataAnalysis', BadDA, raising=False)

    df = pd.DataFrame({'Date': ['2021-01'], 'Data': [1]})
    assert tsf.analysisData(df) == -1

    df2 = pd.DataFrame({'Date': ['2021-01', '2021-02'], 'Data': [1,2]})
    mock_mr = pd.DataFrame({'Model': [], 'Status': []})
    assert tsf.modelResult(df2, mock_mr) == -1


def test_generate_ts_data(session_air_data):
    dataDF, futureDF = tsf.generateTSData(session_air_data, format='%Y-%m', freq='MS', periods=5)
    assert isinstance(dataDF, pd.DataFrame)
    assert isinstance(futureDF, pd.DataFrame)
    assert not dataDF.empty


def test_analysis_data(session_ts_data):
    # analysisData should run on prepared TS data without raising
    tsf.analysisData(session_ts_data)
    assert True


def test_build_model_fast(mock_fast_model, model_data):
    dataDF, futureDF = model_data
    result = tsf.buildModel(dataDF, futureDF, seasonal=12, modelApproach='FAST', testSize=80)
    assert isinstance(result, pd.DataFrame)
    assert 'Model' in result.columns


def test_build_model_best(mock_slow_models, model_data):
    dataDF, futureDF = model_data
    # mock_slow_models patches expensive internals so buildModel should complete quickly
    result = tsf.buildModel(dataDF, futureDF, seasonal=12, modelApproach='BEST', testSize=80)
    # result may be a DataFrame or other truthy summary; assert it did not error
    assert result is not None


def test_end_to_end_quick(session_air_data, mock_fast_model):
    # Quick end-to-end smoke: generate data, analyze, and build FAST model
    dataDF, futureDF = tsf.generateTSData(session_air_data, format='%Y-%m', freq='MS', periods=3)
    _ = tsf.analysisData(dataDF)
    out = tsf.buildModel(dataDF, futureDF, seasonal=12, modelApproach='FAST', testSize=80)
    assert out is not None


def test_modelresult_empty_and_valid(monkeypatch):
    tsf = importlib.import_module('simpliml.timeseriesforecast')
    df = pd.DataFrame({'Date': ['2021-01'], 'Data': [1]})
    empty = pd.DataFrame()
    # empty mdlResult should return -1
    assert tsf.modelResult(df, empty) == -1

    # replace dataAnalysis with dummy that returns a known tuple
    class DummyDA:
        def __init__(self, *a, **k):
            pass
        def modelResult(self, dataDF, mdlResult, modelApproach='Best'):
            return ('ok',)

    monkeypatch.setattr(tsf, 'dataAnalysis', DummyDA, raising=False)
    res = tsf.modelResult(df, pd.DataFrame({'A': [1]}))
    assert res == ('ok',)


def test_modelresult_exception_returns_minus_one(monkeypatch):
    tsf = importlib.import_module('simpliml.timeseriesforecast')
    df = pd.DataFrame({'Date': ['2021-01'], 'Data': [1]})

    class BadDA:
        def __init__(self, *a, **k):
            pass
        def modelResult(self, dataDF, mdlResult, modelApproach='Best'):
            raise RuntimeError('boom')

    monkeypatch.setattr(tsf, 'dataAnalysis', BadDA, raising=False)
    res = tsf.modelResult(df, pd.DataFrame({'A': [1]}))
    assert res == -1

