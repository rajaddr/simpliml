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



def test_dataanalysis_execution_and_modelresult_every_branch():
    da = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')

    df = pd.DataFrame({
        'Date': pd.date_range('2021-01', periods=12, freq='M').astype(str),
        'Data': [100, 101, 99, 102, 104, 103, 105, 107, 106, 108, 110, 109],
    })

    chart = _mock_alt_chart()
    with patch.object(da, 'alt') as mock_alt:
        mock_alt.Chart.return_value = chart
        mock_alt.value = lambda x: x
        mock_alt.OverlayMarkDef = lambda *a, **k: object()
        mock_alt.Tooltip = lambda *a, **k: 'tooltip'
        mock_alt.X = lambda *a, **k: 'X'
        mock_alt.Y = lambda *a, **k: 'Y'
        mock_alt.Color = lambda *a, **k: 'Color'
        mock_alt.Scale = lambda *a, **k: None
        mock_alt.selection_interval = lambda *a, **k: None
        mock_alt.selection = lambda *a, **k: None
        mock_alt.condition = lambda *a, **k: None
        mock_alt.layer = lambda *a, **k: chart
        with patch.object(da, 'plt') as mock_plt, patch.object(da, 'sns') as mock_sns, patch.object(da, 'sm') as mock_sm, patch.object(da, 'seasonal_decompose') as mock_dec, patch.object(da, 'adfuller', return_value=(0.0, 0.01)), patch.object(da, 'jb', return_value=(0.0, 0.2)):
            mock_plt.subplots.return_value = (MagicMock(), MagicMock())
            mock_sm.graphics.tsa.plot_acf = MagicMock()
            mock_sm.graphics.tsa.plot_pacf = MagicMock()
            mock_dec.return_value.plot.return_value.suptitle = MagicMock()
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                d = da.dataAnalysis()
                d.analysisData(df)
            finally:
                sys.stdout = old_stdout

    dataDF = pd.DataFrame({'Date': ['2021-01', '2021-02', '2021-03'], 'Data': [100, 105, 110]})
    best_df = pd.DataFrame({
        'Status': [True, False],
        'MAPE': [1.0, 5.0],
        'Test': [[['2021-01', 100], ['2021-02', 105]], [['2021-01', 99]]],
        'Predict': [[['2021-03', 110]], [['2021-03', 111]]],
        'Model': ['ModelA', 'ModelB'],
        'MAE': [1.0, 2.0],
        'MSE': [2.0, 4.0],
        'RMSE': [1.0, 2.0],
        'RMSLE': [0.1, 0.2],
        'R2': [0.9, 0.8],
        'SMAPE': [1.0, 2.0],
        'MFE': [0.2, 0.5],
        'NMSE': [0.1, 0.2],
        'THEIL_U_STATISTIC': [0.1, 0.2],
    })

    with patch.object(da, 'alt') as mock_alt:
        mock_alt.Chart.return_value = _mock_alt_chart()
        mock_alt.value = lambda x: x
        mock_alt.selection = lambda *a, **k: None
        mock_alt.condition = lambda *a, **k: None
        mock_alt.layer = lambda *a, **k: _mock_alt_chart()
        d = da.dataAnalysis()
        res_best = d.modelResult(dataDF, best_df, 'BEST')
        assert res_best is not None

        res_other = d.modelResult(dataDF, best_df, 'ModelB')
        assert res_other is not None

        bad = d.modelResult(pd.DataFrame({'Date': [], 'Data': []}), pd.DataFrame({'Status': [], 'MAPE': []}), 'ANY')
        assert bad == -1


def test_remaining_dataanalysis_plot_and_display_branches():
    da = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')

    class Chart:
        def __add__(self, other):
            return self

        def mark_line(self, *args, **kwargs):
            return self

        mark_trail = mark_bar = mark_point = mark_text = mark_rule = mark_line

        def encode(self, *args, **kwargs):
            return self

        def properties(self, *args, **kwargs):
            return self

        def interactive(self):
            return self

        def display(self):
            return None

        def add_selection(self, *args, **kwargs):
            return self

        def configure_header(self, *args, **kwargs):
            return self

        def transform_filter(self, *args, **kwargs):
            return self

    class Alt:
        Chart = staticmethod(lambda *args, **kwargs: Chart())
        value = staticmethod(lambda value: value)
        OverlayMarkDef = staticmethod(lambda *args, **kwargs: object())
        Tooltip = staticmethod(lambda *args, **kwargs: object())
        X = staticmethod(lambda *args, **kwargs: object())
        Y = staticmethod(lambda *args, **kwargs: object())
        Color = staticmethod(lambda *args, **kwargs: object())
        Scale = staticmethod(lambda *args, **kwargs: object())
        selection_interval = staticmethod(lambda *args, **kwargs: object())
        selection = staticmethod(lambda *args, **kwargs: object())
        layer = staticmethod(lambda *args, **kwargs: Chart())

    df = pd.DataFrame({
        'Date': pd.date_range('2021-01-01', periods=24, freq='D'),
        'Data': np.arange(100.0, 124.0),
    })
    sm = MagicMock()
    sm.graphics.tsa.plot_acf = MagicMock()
    sm.graphics.tsa.plot_pacf = MagicMock()
    plt = MagicMock()
    plt.subplots.return_value = (MagicMock(), MagicMock())
    decomposition = MagicMock()
    decomposition.plot.return_value.suptitle = MagicMock()

    with patch.object(da, 'alt', Alt), patch.object(da, 'plt', plt), \
            patch.object(da, 'sns', MagicMock()), patch.object(da, 'sm', sm), \
            patch.object(da, 'seasonal_decompose', return_value=decomposition), \
            patch.object(da, 'adfuller', return_value=(np.float64(0.0), np.float64(0.01))), \
            patch.object(da, 'jb', return_value=(np.float64(0.0), np.float64(0.2))):
        da.dataAnalysis().analysisData(df)

    failing_alt = MagicMock()
    failing_alt.layer.side_effect = RuntimeError('display failure')
    with patch.object(da, 'alt', failing_alt):
        result = da.dataAnalysis().modelResult(
            pd.DataFrame({'Date': ['2021-01'], 'Data': [1.0]}),
            pd.DataFrame({
                'Status': [True], 'MAPE': [1.0], 'Test': [[['2021-01', 1.0]]],
                'Predict': [[['2021-02', 2.0]]], 'Model': ['A'],
                'MAE': [1.0], 'MSE': [1.0], 'RMSE': [1.0], 'RMSLE': [1.0],
                'R2': [1.0], 'SMAPE': [1.0], 'MFE': [1.0], 'NMSE': [1.0],
                'THEIL_U_STATISTIC': [1.0],
            }),
            'BEST',
        )
    assert result != -1


def test_dataanalysis_analysis_plots_execute():
    """Test dataAnalysis.analysisData to execute all plot lines (120-154, 167)"""
    da = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')
    
    # Create proper data
    df = pd.DataFrame({
        'Date': pd.date_range('2021-01', periods=24, freq='M'),
        'Data': np.sin(np.linspace(0, 4*np.pi, 24)) * 50 + 100
    })
    df['Date'] = df['Date'].astype(str)
    
    # Mock display functions to avoid interactive plotting but still execute the code
    class MockChart:
        def __init__(self, *a, **k):
            pass
        def mark_line(self, *a, **k):
            return self
        def mark_trail(self, *a, **k):
            return self
        def mark_bar(self, *a, **k):
            return self
        def mark_point(self, *a, **k):
            return self
        def encode(self, *a, **k):
            return self
        def properties(self, *a, **k):
            return self
        def interactive(self):
            return self
        def display(self):
            return None
        def add_selection(self, *a, **k):
            return self
        def configure_header(self, *a, **k):
            return self
        def transform_filter(self, *a, **k):
            return self
    
    with patch.object(da, 'alt') as mock_alt:
        mock_alt.Chart = MockChart
        mock_alt.value = lambda x: x
        mock_alt.OverlayMarkDef = lambda: None
        mock_alt.X = lambda *a, **k: "X"
        mock_alt.Y = lambda *a, **k: "Y"
        mock_alt.Color = lambda *a, **k: "Color"
        mock_alt.Scale = lambda *a, **k: {}
        mock_alt.Tooltip = lambda *a: "Tooltip"
        mock_alt.selection_interval = lambda *a, **k: None
        mock_alt.selection = lambda *a, **k: None
        mock_alt.condition = lambda *a, **k: None
        mock_alt.layer = lambda *a: MockChart()
        
        with patch.object(da, 'plt'):
            with patch.object(da, 'sns'):
                with patch.object(da, 'sm'):
                    with patch.object(da, 'seasonal_decompose'):
                        with patch.object(da, 'adfuller', return_value=(0.0, 0.01)):
                            with patch.object(da, 'jb', return_value=(0.0, 0.2)):
                                # Capture output
                                old_stdout = sys.stdout
                                sys.stdout = StringIO()
                                try:
                                    d = da.dataAnalysis()
                                    d.analysisData(df)
                                finally:
                                    sys.stdout = old_stdout


def test_dataanalysis_modelresult_best_approach():
    """Test modelResult with BEST approach to hit line 162-165"""
    da = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')
    
    # Create mock data for modelResult
    dataDF = pd.DataFrame({
        'Date': ['2021-01', '2021-02', '2021-03'],
        'Data': [100, 105, 110]
    }).set_index(pd.Index(['2021-01', '2021-02', '2021-03'], name='Date'))
    
    mdlResult = pd.DataFrame({
        'Status': [True, True],
        'MAPE': [5.0, 3.0],  # Second one is min
        'Test': [[['2021-01', 100], ['2021-02', 105]], [['2021-01', 101], ['2021-02', 106]]],
        'Predict': [[['2021-03', 110]], [['2021-03', 111]]],
        'Model': ['Model1', 'Model2'],
        'MAE': [1.0, 0.5],
        'MSE': [2.0, 1.0],
        'RMSE': [1.4, 1.0],
        'RMSLE': [0.1, 0.05],
        'R2': [0.9, 0.95],
        'SMAPE': [1.0, 0.5],
        'MFE': [0.1, 0.05],
        'NMSE': [0.1, 0.05],
        'THEIL_U_STATISTIC': [0.1, 0.05]
    })
    
    with patch.object(da, 'alt') as mock_alt:
        mock_alt.Chart = MagicMock(return_value=MagicMock(
            mark_line=MagicMock(return_value=MagicMock(
                encode=MagicMock(return_value=MagicMock(
                    properties=MagicMock(return_value=MagicMock(
                        interactive=MagicMock(return_value=MagicMock(display=MagicMock()))
                    ))
                ))
            ))
        ))
        mock_alt.selection = MagicMock(return_value=None)
        mock_alt.value = lambda x: x
        mock_alt.condition = lambda *a: None
        
        with patch.object(da, 'plt'):
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                d = da.dataAnalysis()
                # Test BEST approach (line 162-165)
                res = d.modelResult(dataDF, mdlResult, 'BEST')
                assert res is not None
            finally:
                sys.stdout = old_stdout


def test_dataanalysis_modelresult_other_approach():
    """Test modelResult with non-BEST approach to hit line 167"""
    da = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')
    
    dataDF = pd.DataFrame({
        'Date': ['2021-01', '2021-02'],
        'Data': [100, 105]
    }).set_index(pd.Index(['2021-01', '2021-02'], name='Date'))
    
    mdlResult = pd.DataFrame({
        'Status': [True],
        'MAPE': [5.0],
        'Test': [[['2021-01', 100]]],
        'Predict': [[['2021-02', 105]]],
        'Model': ['RandomForest'],
        'MAE': [1.0],
        'MSE': [2.0],
        'RMSE': [1.4],
        'RMSLE': [0.1],
        'R2': [0.9],
        'SMAPE': [1.0],
        'MFE': [0.1],
        'NMSE': [0.1],
        'THEIL_U_STATISTIC': [0.1]
    })
    
    with patch.object(da, 'alt') as mock_alt:
        mock_alt.Chart = MagicMock(return_value=MagicMock(
            mark_line=MagicMock(return_value=MagicMock(
                encode=MagicMock(return_value=MagicMock(
                    properties=MagicMock(return_value=MagicMock(
                        interactive=MagicMock(return_value=MagicMock(display=MagicMock()))
                    ))
                ))
            ))
        ))
        mock_alt.selection = MagicMock(return_value=None)
        mock_alt.value = lambda x: x
        mock_alt.condition = lambda *a: None
        
        with patch.object(da, 'plt'):
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                d = da.dataAnalysis()
                # Test non-BEST approach (line 167)
                res = d.modelResult(dataDF, mdlResult, 'RandomForest')
                assert res is not None
            finally:
                sys.stdout = old_stdout


def test_dataanalysis_modelresult_exception_paths():
    """Test exception paths in modelResult (lines 243-244, 248-250)"""
    da = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')
    
    # Empty or malformed data to trigger exceptions
    dataDF = pd.DataFrame({'Date': [], 'Data': []})
    mdlResult = pd.DataFrame({'Status': [], 'MAPE': []})
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        d = da.dataAnalysis()
        # This should hit exception paths and return -1
        res = d.modelResult(dataDF, mdlResult, 'ANY')
        assert res == -1
    finally:
        sys.stdout = old_stdout


def test_dataanalysis_analysis_and_modelresult(monkeypatch):
    # Patch plotting and heavy functions to no-op
    class DummyChart:
        def __init__(self, *args, **kwargs):
            pass
        def mark_line(self, *a, **k): return self
        def encode(self, *a, **k): return self
        def mark_trail(self, *a, **k): return self
        def mark_bar(self, *a, **k): return self
        def mark_text(self, *a, **k): return self
        def add_selection(self, *a, **k): return self
        def properties(self, *a, **k): return self
        def interactive(self): return self
        def display(self): return None

    import importlib
    da_module = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')
    import numpy as _np
    class DummyChart:
        def __init__(self, *args, **kwargs):
            pass
        def mark_line(self, *a, **k): return self
        def mark_point(self, *a, **k): return self
        def mark_trail(self, *a, **k): return self
        def mark_bar(self, *a, **k): return self
        def mark_text(self, *a, **k): return self
        def mark_rule(self, *a, **k): return self
        def encode(self, *a, **k): return self
        def add_selection(self, *a, **k): return self
        def properties(self, *a, **k): return self
        def interactive(self): return self
        def display(self): return None
        def configure_header(self, *a, **k): return self
        def transform_filter(self, *a, **k): return self
        def __add__(self, other):
            return self
        def __radd__(self, other):
            return self

    class DummyAlt:
        Chart = DummyChart
        @staticmethod
        def value(v):
            return v
        @staticmethod
        def OverlayMarkDef(*a, **k):
            return None
        @staticmethod
        def selection(*a, **k):
            # return a dummy selector object
            return type('Selector', (), {})()
        @staticmethod
        def selection_interval(*a, **k):
            return None
        @staticmethod
        def condition(nearest, a, b):
            # return a representation used by encode; prefer the second arg if string
            return a if isinstance(a, str) else b
        @staticmethod
        def layer(*args, **kwargs):
            return DummyChart()
        class Scale:
            def __init__(self, *a, **k):
                pass
        class Color:
            def __init__(self, *a, **k):
                pass

    monkeypatch.setattr(da_module, 'alt', DummyAlt)
    monkeypatch.setattr(da_module.plt, 'show', lambda *a, **k: None)
    monkeypatch.setattr(da_module.sns, 'boxplot', lambda *a, **k: None)
    monkeypatch.setattr(da_module.sns, 'swarmplot', lambda *a, **k: None)
    monkeypatch.setattr(da_module.sns, 'distplot', lambda *a, **k: None)
    # adfuller/jb return values must support .round() used in code -> return numeric at index 0 and pvalue at index1
    monkeypatch.setattr(da_module, 'adfuller', lambda x: (_np.float64(0.0), _np.float64(0.01)))
    monkeypatch.setattr(da_module, 'jb', lambda x: (_np.float64(0.0), _np.float64(0.2)))
    monkeypatch.setattr(da_module, 'seasonal_decompose', lambda *a, **k: type('D', (), {'plot': lambda self: None})())
    # run - robust instantiation whether module/class imported
    import importlib
    da_module = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')
    if hasattr(da_module, 'dataAnalysis'):
        da = da_module.dataAnalysis()
    elif isinstance(da_module, type):
        da = da_module()
    else:
        da = da_mod.dataAnalysis()
    sample = pd.DataFrame({'Date': ['2021-01', '2021-02', '2021-03'], 'Data': [100, 110, 105]})
    da.analysisData(sample.copy())

    # Test modelResult path
    # Create dummy model result
    mock_mr = pd.DataFrame({'Model': ['M1'], 'Status': [True], 'MAPE': [0.1],
                            'MAE': [1], 'MSE': [1], 'RMSE': [1], 'RMSLE': [0.1], 'R2': [0.9],
                            'SMAPE': [0.05], 'MFE': [0.1], 'NMSE': [0.01], 'THEIL_U_STATISTIC': [0.02],
                            'Test': [[["2021-01", 100]]], 'Predict': [[["2021-02", 105]]]
                            })
    dataDF = sample.copy()
    dataDF.columns = ['Date', 'Data']
    val = da.modelResult(dataDF.set_index('Date').reset_index(), mock_mr, 'BEST')
    assert isinstance(val, tuple)


def test_dataanalysis_analysis_minimal_execution():
    """Test analysisData with minimal setup"""
    da = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')
    
    # Create minimal data
    df = pd.DataFrame({
        'Date': ['2021-01', '2021-02', '2021-03'],
        'Data': [100, 105, 110]
    })
    
    with patch.object(da, 'alt'):
        with patch.object(da, 'plt'):
            with patch.object(da, 'sns'):
                with patch.object(da, 'sm'):
                    with patch.object(da, 'seasonal_decompose'):
                        with patch.object(da, 'adfuller', return_value=(0.0, 0.01)):
                            with patch.object(da, 'jb', return_value=(0.0, 0.2)):
                                old_stdout = sys.stdout
                                sys.stdout = StringIO()
                                try:
                                    d = da.dataAnalysis()
                                    d.analysisData(df)
                                finally:
                                    sys.stdout = old_stdout


def test_dataanalysis_modelresult_with_valid_data():
    """Test modelResult with valid structured data"""
    da = importlib.import_module('simpliml.timeseriesforecast.dataAnalysis')
    
    dataDF = pd.DataFrame({
        'Date': pd.date_range('2021-01', periods=3),
        'Data': [100, 105, 110]
    }).set_index(pd.Index(pd.date_range('2021-01', periods=3), name='Date'))
    
    mdlResult = pd.DataFrame({
        'Status': [True, False],
        'MAPE': [2.0, 5.0],
        'Test': [[['2021-01', 100]], [['2021-01', 101]]],
        'Predict': [[['2021-02', 105]], [['2021-02', 106]]],
        'Model': ['Model1', 'Model2'],
        'MAE': [1.0, 2.0],
        'MSE': [1.0, 4.0],
        'RMSE': [1.0, 2.0],
        'RMSLE': [0.1, 0.2],
        'R2': [0.9, 0.8],
        'SMAPE': [1.0, 2.0],
        'MFE': [0.5, 1.0],
        'NMSE': [0.1, 0.2],
        'THEIL_U_STATISTIC': [0.1, 0.2]
    })
    
    with patch.object(da, 'alt') as mock_alt:
        mock_alt.Chart = MagicMock(return_value=MagicMock(
            mark_line=MagicMock(return_value=MagicMock(
                encode=MagicMock(return_value=MagicMock(
                    properties=MagicMock(return_value=MagicMock(
                        interactive=MagicMock(return_value=MagicMock(display=MagicMock())),
                        add_selection=MagicMock(return_value=MagicMock(display=MagicMock())),
                    ))
                ))
            )),
            mark_point=MagicMock(return_value=MagicMock(
                encode=MagicMock(return_value=MagicMock(
                    add_selection=MagicMock(return_value=MagicMock())
                ))
            )),
            mark_text=MagicMock(return_value=MagicMock(
                encode=MagicMock(return_value=MagicMock())
            )),
            mark_rule=MagicMock(return_value=MagicMock(
                encode=MagicMock(return_value=MagicMock(
                    transform_filter=MagicMock(return_value=MagicMock())
                ))
            )),
        ))
        mock_alt.selection = MagicMock(return_value=None)
        mock_alt.value = lambda x: x
        mock_alt.condition = lambda *a, **k: None
        mock_alt.layer = MagicMock(return_value=MagicMock(
            properties=MagicMock(return_value=MagicMock(
                interactive=MagicMock(return_value=MagicMock(display=MagicMock()))
            ))
        ))
        
        with patch.object(da, 'plt'):
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                d = da.dataAnalysis()
                res = d.modelResult(dataDF, mdlResult, 'BEST')
                assert res is not None
            finally:
                sys.stdout = old_stdout


def _mock_alt_chart():
    chart = MagicMock()
    chart.mark_line.return_value = chart
    chart.mark_trail.return_value = chart
    chart.mark_bar.return_value = chart
    chart.mark_point.return_value = chart
    chart.mark_text.return_value = chart
    chart.mark_rule.return_value = chart
    chart.encode.return_value = chart
    chart.properties.return_value = chart
    chart.interactive.return_value = chart
    chart.display.return_value = None
    chart.add_selection.return_value = chart
    chart.configure_header.return_value = chart
    chart.transform_filter.return_value = chart
    return chart

