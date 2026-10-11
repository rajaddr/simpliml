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



def test_tsfcommon_eval_and_split():
    y = pd.Series([100.0, 110.0, 105.0])
    preds = pd.Series([99.0, 111.0, 104.0])
    # support importing the class or module
    if hasattr(common_mod, 'tsfCommon'):
        cm = common_mod.tsfCommon()
    elif isinstance(common_mod, type):
        cm = common_mod()
    else:
        cm = common_mod.tsfCommon()
    res = cm.modelEval(y, preds)
    assert isinstance(res, list) and len(res) == 10

    df = pd.DataFrame({'A': [1, 2, 3, 4]})
    train, test = cm.getTrainTest(df, 0.5)
    assert len(train) == 2 and len(test) == 2

