import pytest, pandas as pd, simpliml.timeseriesforecast as tsf, warnings
warnings.filterwarnings("ignore")

def test_add():
    sourceDF = pd.read_csv("tests/AirPassengers.csv")
    dataDF, futureDF = tsf.generateTSData(sourceDF, format='%Y-%m', freq='MS', periods=30)
    tsf.analysisData(dataDF)
    mdlResult = tsf.runModel(dataDF, futureDF, seasonal=12, modelApproach='FAST', testSize=80)
    mdlResult1 = tsf.runThreadModel(dataDF, futureDF, seasonal=12, modelApproach='BEST', testSize=80)
    mdlResult2 = tsf.runProcessModel(dataDF, futureDF, seasonal=12, modelApproach='FAST', testSize=80)
    mdlOutPut = tsf.modelResult(dataDF, mdlResult, modelApproach='Best')
    mdlOutPut1 = tsf.modelResult(dataDF, mdlResult, modelApproach='HOLT')