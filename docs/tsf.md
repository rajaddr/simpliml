# Time Series
- **Pre-built Models:** Includes popular time series forecasting models like ARIMA, SARIMA, torch, and more.
- **Seamless Integration:** Load data, preprocess, and run forecasting models in one place.
- **Automatic Forecasting:** Automatically generates forecasts for future time steps once a model is selected.
- **Visualization:** Built-in tools for visualizing both historical data and forecasted values.
- **Customizable:** Fine-tune model parameters to suit your specific use case.
- **Extensive Documentation:** Detailed guides and examples to help you get started.


#### Import SimpliML Time Series
```python
import pandas as pd
import simpliml.timeseriesforecast as tsf
```

#### Build Time Series Data
```python
sourceDF = pd.read_csv("") # Any Input data read
dataDF, futureDF = tsf.generateTSData(sourceDF, format='%Y-%m', freq='MS', periods=30)
```
Parameters:-

  - format : str, optional
    - Please refer [strftime-and-strptime-behavior](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior)
  - freq : str, optional
    - Please refer [timeseries-offset-aliases](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases)
  - periods : int, optional
    - Forcasting time period

#### Analysis Data
```python
tsf.analysisData(dataDF) # This will work only in interactive computational environment like Jupyter Notebook/lab/hub ..etc 
```

#### Build Model and forcast
```python
mdlResult = tsf.runModel(dataDF, futureDF, seasonal=12, modelApproach = 'FAST', testSize=80)  # Single Process Thread
(OR)
mdlResult = tsf.runThreadModel(dataDF, futureDF, seasonal=12, modelApproach = 'FAST', testSize=80) # Single Process Multiple Thread (Thread : CPU Count * 2)
(OR)
mdlResult = tsf.runProcessModel(dataDF, futureDF, seasonal=12, modelApproach = 'FAST', testSize=80) # Multiple Process (Process : CPU Count / 4) # Advise to use only in Windows  
```
Parameters:-

  - seasonal : int, optional
    - The number of periods in a complete seasonal cycle, 
    - Example 
      -  1 : Yearly data
      -  2 : Half-yearly data
      -  4 : Quarterly data 
      -  7 : Daily data with a weekly cycle
      - 52 : Weekly Data
  - modelApproach : ["BEST", "FAST"], optional
    - BEST : Best model build with multiple permutation and combination
    - FAST : Fast model build with limited permutation and combination
  - testSize : int, optional
    - Test Size by defult 80:20 rule


#### Model Result Analysis
```python
mdlOutPut = tsf.modelResult(dataDF, mdlResult, modelApproach='Best') 
```
Parameters:-
  - modelApproach : str, optional
    - BEST MAPE analysis report and can pass the model name, get the analysis report