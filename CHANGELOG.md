
## 1.5.0
  - Improving PyTorch LSTM input handling and device compatibility
  - Expanding automated test coverage to 100%
  - Adding coverage tests for data analysis and forecasting model branches
  - Suppressing expected scikit-learn search warnings in tests

## 1.4.0 
  - Fixing min/max date in generateTSData
  - Adding logging.getLogger function in dataModeling

## 1.3.0 
  - Removing auto_arima (pmdarima)

## 1.2.0 
  - Adding Documentation
  - Adding pytest
  - Bug fix 

## 1.1.0
  - Fixing the library directory structure    

## 1.0.0
  - Initial Version
    - Time Series Analysis
      - statsmodels
        - ARIMA
        - AutoReg
        - ExponentialSmoothing
        - Holt
        - SARIMAX
        - SimpleExpSmoothing
        - VAR
        - VARMAX
      - pmdarima
        - auto_arima
      - scikit-learn
        - BaseEnsemble
        - RandomForestClassifier
        - RandomForestRegressor
        - ExtraTreesClassifier
        - ExtraTreesRegressor
        - BaggingClassifier
        - BaggingRegressor
        - IsolationForest
        - GradientBoostingClassifier
        - GradientBoostingRegressor
        - AdaBoostClassifier
        - AdaBoostRegressor
        - VotingClassifier
        - VotingRegressor
        - StackingClassifier
        - StackingRegressor
        - HistGradientBoostingClassifier
        - HistGradientBoostingRegressor
      - PyTorch
        - Tensor
