<p align="center">
<img src="https://i.ibb.co/jvM4fsVN/Simpli-ML.png"/>
</p><div align="center">

![Python](https://img.shields.io/pypi/pyversions/simpliml/1.5.0?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=python)
[![Version](https://img.shields.io/static/v1?label=Version&labelColor=007676&message=1.5.0&color=01C0C0&style=flat)](https://pypi.org/project/simpliml/1.5.0/)
![PyPI - Version](https://img.shields.io/pypi/v/simpliml?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=pypi)
![Package Status](https://img.shields.io/static/v1?label=Status&labelColor=007676&message=Planning&color=01C0C0&style=flat)
[![License](https://img.shields.io/static/v1?label=License&labelColor=007676&message=MIT&color=01C0C0&style=flat)](https://github.com/rajaddr/simpliml/blob/master/LICENSE)
![PyPI - Downloads](https://img.shields.io/pypi/dm/simpliml?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=pypi)
[![Documentation Status](https://img.shields.io/readthedocs/simpliml?style=flat&labelColor=007676&color=01C0C0&logoColor=01C0C0&logo=readthedocs)](https://simpliml.readthedocs.io/en/latest/?badge=latest)
![Test Coverage](https://img.shields.io/static/v1?label=TestCoverage&labelColor=007676&message=100%&color=01C0C0&style=flat&logoColor=01C0C0&logo=pytest)

<hr>
</div>

**SimpliML** is a Python machine-learning library designed to simplify the
data lifecycle. The project brings data preparation, exploratory analysis,
forecasting, model evaluation, and result interpretation together in one
consistent workflow. It is intended for data scientists, analysts, students,
and developers who want practical forecasting tools without building the
entire modeling pipeline from scratch.

SimpliML currently focuses on time-series forecasting. It accepts historical
observations, prepares regular date-based datasets, evaluates multiple model
families, and returns comparable results for selecting a forecast. The package
is designed to work with familiar Python data tools, especially pandas
`DataFrame` objects.

## Key Features

- **Time-Series Data Preparation**  
  Convert date columns, generate regular date ranges, preserve historical
  observations, handle missing dates, add calendar features, and create future
  periods for prediction.

- **Data Analysis**  
  Explore trends, seasonality, distributions, autocorrelation, and partial
  autocorrelation with interactive and statistical visualizations.

- **Multiple Model Families**  
  Train statistical models through statsmodels, ensemble and regression models
  through scikit-learn, and neural-network forecasting models through PyTorch.

- **Flexible Execution**  
  Run model evaluation sequentially, with multiple threads, or with multiple
  processes depending on the workload and environment.

- **Model Comparison**  
  Compare successful models using MAE, MSE, RMSE, RMSLE, MAPE, R2, SMAPE,
  MFE, NMSE, and Theil's U statistic.

- **Forecast Result Review**  
  Select the best available model by MAPE or inspect a specific model, then
  combine historical, test, and forecast values for analysis and visualization.

## Forecasting Workflow

The package follows a simple end-to-end workflow:

1. **Prepare** historical date and target-value data.
2. **Generate** a regular time series and future dates.
3. **Analyze** trends, seasonality, and statistical behavior.
4. **Train** supported forecasting models with a selected seasonal period.
5. **Evaluate** model performance on held-out observations.
6. **Select** the best successful model and review its forecast.

The public time-series API includes `generateTSData`, `analysisData`,
`runModel`, `runThreadModel`, `runProcessModel`, `buildModel`, and
`modelResult`.

## Supported Data and Models

Input data is represented with pandas dataframes. The forecasting workflow uses
a date column and a numeric target column named `Data` after preparation.
Regular data frequencies such as daily, monthly, and quarterly observations
are supported.

The model pipeline can include:

- Statistical forecasting models from statsmodels, including ARIMA, AutoReg,
  exponential smoothing, Holt, SARIMA/SARIMAX, VAR, and VARMAX.
- Scikit-learn regression and ensemble estimators for feature-based
  forecasting.
- PyTorch LSTM-based forecasting models for neural-network workflows.

Model execution can be configured with `FAST` for a reduced search or `BEST`
for a broader search. The selected seasonal period should match the data
frequency, such as `7` for weekly seasonality in daily data or `12` for yearly
seasonality in monthly data.

## Intended Uses

SimpliML can support demand and sales forecasting, operational planning,
monthly or daily KPI prediction, time-series experimentation, classroom
projects, and rapid comparisons of statistical and machine-learning models.
It is a forecasting toolkit rather than a replacement for domain validation,
data quality checks, or production monitoring.

## Documentation

The official documentation is available on
[Read the Docs](https://simpliml.readthedocs.io/).

## Project Links

- [PyPI](https://pypi.org/project/simpliml/)
- [GitHub Repository](https://github.com/rajaddr/simpliml)
- [Issue Tracker](https://github.com/rajaddr/simpliml/issues)

