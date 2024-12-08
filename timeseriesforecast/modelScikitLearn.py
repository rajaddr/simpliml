import pandas as pd, numpy as np, math, time, itertools, inspect, warnings, os, logging
import sklearn.ensemble as sl
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from concurrent import futures

warnings.filterwarnings("ignore")

from .tsfCommon import *


class modelScikitLearn:
    __all__ = [elem for elem in sl.__all__ if elem not in ['RandomTreesEmbedding']]

    def __init__(self, modelName):
        self.modelName = modelName

    def runmodelEnsemble(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        pkg = self.modelName
        dfSize = x_train.shape[0] + x_test.shape[0]
        furSize = x_test.shape[0]
        try:
            argsList = []
            argsSel = {}
            estimatorsSt = 0

            for x in list(inspect.signature(eval("sl." + pkg + ".__init__")).parameters.keys()):
                argsList.append(x)

            if 'estimators' in argsList: estimatorsSt = 1
            if 'n_estimators' in argsList: argsSel['n_estimators'] = [int(x) for x in np.linspace(start=20, stop=200,
                                                                                                  num=round(math.log(
                                                                                                      dfSize)))]
            if 'max_features' in argsList: argsSel['max_features'] = list(
                set([round(math.log2(furDF.shape[0])), round(math.sqrt(furDF.shape[0])), round(furDF.shape[0])]))
            if 'max_depth' in argsList: argsSel['max_depth'] = [int(x) for x in
                                                                np.linspace(1, 10, num=round(math.log(dfSize)))]
            if 'min_samples_split' in argsList: argsSel['min_samples_split'] = [int(x) for x in np.linspace(1, 10,
                                                                                                            num=round(
                                                                                                                math.log(
                                                                                                                    dfSize)))]
            if 'min_samples_leaf' in argsList: argsSel['min_samples_leaf'] = [int(x) for x in np.linspace(1, 10,
                                                                                                          num=round(
                                                                                                              math.log(
                                                                                                                  dfSize)))]
            if 'bootstrap' in argsList: argsSel['bootstrap'] = [True, False]
            try:
                modelSel = eval("sl.{0}()".format(str(pkg)))
                if modelApproach == 'FAST':
                    rfr_random = RandomizedSearchCV(estimator=modelSel, param_distributions=argsSel, n_iter=120,
                                                    scoring='neg_mean_absolute_error', cv=2, verbose=0, random_state=42,
                                                    n_jobs=-1, return_train_score=True).fit(x_train, y_train)
                else:
                    rfr_random = GridSearchCV(estimator=modelSel, param_grid=argsSel, scoring='neg_mean_absolute_error',
                                              cv=2, verbose=0, n_jobs=-1, return_train_score=True).fit(x_train, y_train)
                testData = eval("sl." + str(pkg) + "(**rfr_random.best_params_).fit(x_train, y_train).predict(x_test)")
                furData = eval("sl." + str(
                    pkg) + "(**rfr_random.best_params_).fit(pd.concat([x_train, x_test],axis=0), pd.concat([y_train, y_test],axis=0)).predict(furDF)")
                modelSt = tsfCommon().modelEval(y_test, testData)
                rtVal.append(["sklearn.ensemble", pkg, True, modelSt[0], modelSt[1], modelSt[2], modelSt[3], modelSt[4],
                              modelSt[5], modelSt[6], modelSt[7], modelSt[8], modelSt[9],
                              pd.concat([y_test.reset_index(), pd.DataFrame(testData)], axis=1)[
                                  ['Date', 0]].values.tolist(),
                              pd.concat([furDF.reset_index(), pd.DataFrame(furData)], axis=1)[
                                  ['Date', 0]].values.tolist()])
            except Exception as e:
                rtVal.append(
                    ["sklearn.ensemble", pkg, False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []])
        except Exception as e:
            print(e)
        return rtVal


if __name__ == '__main__':
    print("Main")
