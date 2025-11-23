import pytest
from unittest.mock import MagicMock, patch
import pandas as pd

# Import the functions to test
from simpliml.timeseriesforecast import runProcessModel, buildModel, modelResult

@pytest.fixture
def sample_dfs():
    dataDF = pd.DataFrame({'a': [1, 2, 3]})
    furDF = pd.DataFrame({'b': [4, 5, 6]})
    return dataDF, furDF

@patch('simpliml.timeseriesforecast.__init__.dataModeling')
def test_runProcessModel_success(mock_dataModeling, sample_dfs):
    dataDF, furDF = sample_dfs
    mock_instance = MagicMock()
    mock_instance.buildModel.return_value = 'model_result'
    mock_dataModeling.return_value = mock_instance
    result = runProcessModel(dataDF, furDF)
    assert result == 'model_result'

@patch('simpliml.timeseriesforecast.__init__.dataModeling')
def test_buildModel_success(mock_dataModeling, sample_dfs):
    dataDF, furDF = sample_dfs
    mock_instance = MagicMock()
    mock_instance.buildModel.return_value = 'model_result'
    mock_dataModeling.return_value = mock_instance
    result = buildModel(dataDF, furDF)
    assert result == 'model_result'

@patch('simpliml.timeseriesforecast.__init__.dataAnalysis')
def test_modelResult_success(mock_dataAnalysis, sample_dfs):
    dataDF, _ = sample_dfs
    mdlResult = pd.DataFrame({'c': [7, 8]})
    mock_instance = MagicMock()
    mock_instance.modelResult.return_value = 'analysis_result'
    mock_dataAnalysis.return_value = mock_instance
    result = modelResult(dataDF, mdlResult)
    assert result == 'analysis_result'

def test_runProcessModel_invalid_shape(sample_dfs):
    dataDF, furDF = pd.DataFrame({'a': [1]}), pd.DataFrame({'b': []})
    result = runProcessModel(dataDF, furDF)
    assert result == -1

def test_buildModel_invalid_shape(sample_dfs):
    dataDF, furDF = pd.DataFrame({'a': [1]}), pd.DataFrame({'b': []})
    result = buildModel(dataDF, furDF)
    assert result == -1

def test_modelResult_invalid_shape(sample_dfs):
    dataDF, _ = sample_dfs
    mdlResult = pd.DataFrame({'c': []})
    result = modelResult(dataDF, mdlResult)
    assert result == -1

