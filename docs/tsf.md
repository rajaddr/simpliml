# Time-series forecasting

SimpliML provides a complete workflow for preparing time-series data, reviewing
its statistical characteristics, running several forecasting models, and
visualizing model results.

## Import the time-series API

```python
import pandas as pd
import simpliml.timeseriesforecast as tsf
```

## Prepare the input data

`generateTSData` expects a two-column `DataFrame`. The first column is parsed as
the date and the second column is the numeric target series. The returned
frames contain the historical data and the requested future dates.

```python
source_df = pd.DataFrame({
    "Date": ["2021-01", "2021-02", "2021-03", "2021-04"],
    "Data": [100, 110, 105, 115],
})

data_df, future_df = tsf.generateTSData(
    source_df,
    format="%Y-%m",
    freq="MS",
    periods=6,
)
```

| Parameter | Description |
| --- | --- |
| `dataDF` | Two-column input frame: date and target value. |
| `format` | Python date format used to parse and format dates. See the [strftime documentation](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior). |
| `freq` | Pandas date frequency, such as `D`, `MS`, or `QS`. See [pandas offset aliases](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases). |
| `periods` | Number of future periods to generate. |

For reliable model training, use a regular frequency and enough historical
observations for the selected seasonal cycle.

## Analyze the time series

`analysisData` calculates summary statistics and displays interactive Altair
charts and statistical plots. Run it in an environment that supports plotting,
such as Jupyter Notebook, JupyterLab, or another graphical Python session.

```python
tsf.analysisData(data_df)
```

The input must contain at least two rows and a `Data` column. The function
returns `-1` for invalid input.

## Train forecasting models

Use one of the following functions:

```python
# One worker
result_df = tsf.runModel(
    data_df,
    future_df,
    seasonal=12,
    modelApproach="FAST",
    testSize=80,
)

# Multiple threads
result_df = tsf.runThreadModel(
    data_df,
    future_df,
    seasonal=12,
    modelApproach="FAST",
    testSize=80,
)

# Multiple processes; use this entry point carefully on Windows
result_df = tsf.runProcessModel(
    data_df,
    future_df,
    seasonal=12,
    modelApproach="FAST",
    testSize=80,
)
```

`buildModel` provides the same workflow with an explicit execution mode and
optional logger name:

```python
result_df = tsf.buildModel(
    data_df,
    future_df,
    seasonal=12,
    modelApproach="BEST",
    testSize=80,
    runType="Single",
)
```

### Model parameters

| Parameter | Description |
| --- | --- |
| `seasonal` | Number of observations in one seasonal cycle. Examples: `7` for daily data with weekly seasonality, `12` for monthly data with yearly seasonality, and `4` for quarterly data with yearly seasonality. |
| `modelApproach` | `FAST` runs a reduced search. `BEST` runs the broader model search and selects the best available result. |
| `testSize` | Percentage of historical observations reserved for testing. The default is `80`, leaving approximately 20% for testing. |
| `runType` | For `buildModel`, use `Single`, `Thread`, or `Process`. |

The model runner evaluates supported statsmodels, scikit-learn, and PyTorch
models. Results are returned as a `DataFrame` with model status, error metrics,
test predictions, future predictions, and runtime.

## Inspect model results

`modelResult` selects the model with the lowest MAPE among successful models
when `modelApproach="BEST"`. Alternatively, pass the exact model name from the
`Model` column in the result frame to inspect that model. It returns combined
data for plotting and a metrics table.

```python
best_data, best_metrics = tsf.modelResult(
    data_df,
    result_df,
    modelApproach="BEST",
)
```

To inspect a specific model:

```python
model_name = result_df.loc[result_df["Status"], "Model"].iloc[0]
model_data, model_metrics = tsf.modelResult(
    data_df,
    result_df,
    modelApproach=model_name,
)
```

An empty result frame or invalid input returns `-1`.

## End-to-end example

```python
import pandas as pd
import simpliml.timeseriesforecast as tsf

source_df = pd.read_csv("monthly_sales.csv")
data_df, future_df = tsf.generateTSData(
    source_df,
    format="%Y-%m",
    freq="MS",
    periods=12,
)

tsf.analysisData(data_df)

result_df = tsf.runModel(
    data_df,
    future_df,
    seasonal=12,
    modelApproach="FAST",
    testSize=80,
)

forecast_data, metrics = tsf.modelResult(
    data_df,
    result_df,
    modelApproach="BEST",
)
```
