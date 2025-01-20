import pytest, pandas as pd, simpliml.timeseriesforecast as tsf, warnings
warnings.filterwarnings("ignore")

def test_add():
    sourceDF = pd.read_csv("tests/AirPassengers.csv")
    dataDF, futureDF = tsf.generateTSData(sourceDF, format='%Y-%m', freq='MS', periods=30)
    #mdlResult = tsf.runModel(dataDF, futureDF, seasonal=12, modelApproach='FAST', testSize=80)