import pandas as pd, numpy as np, os, logging, sys, inspect, itertools, datetime
from concurrent import futures
from .tsfCommon import *
from .modelStatsModels import *
from .modelScikitLearn import *
from .modelPyTorch import *


class dataModeling:
    def submitModel(self, modelName, seasonal, modelApproach, x_train, x_test, y_train, y_test, furDF):
        try:
            logging.info("Model Running :: {0}".format(modelName))
            start_time = time.time()
            return [eval(modelName + "(seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach)")[0] + [
                (time.time() - start_time)]]
        except Exception as e:
            print(e)
        finally:
            logging.info("Model Completed :: {0}".format(modelName))

    def buildModel(self, dataDF, furDF, seasonal, modelApproach, testSize, runType):
        try:
            columnsName = ['Library', 'Model', 'Status', 'MAE', 'MSE', 'RMSE', 'RMSLE', 'MAPE', 'R2', 'SMAPE', 'MFE',
                           'NMSE', 'THEIL_U_STATISTIC', 'Test', 'Predict', 'TT']
            dataDF['Data'] = dataDF['Data'].fillna(dataDF['Data'].mean())
            dataDF = dataDF.set_index('Date')
            furDF = furDF.set_index('Date')
            train, test = tsfCommon().getTrainTest(dataDF, testSize)
            x_train = train.drop('Data', axis=1)
            x_test = test.drop('Data', axis=1)
            y_train = train['Data']
            y_test = test['Data']

            dataVal = []
            mList = []
            [mList.append("modelStatsModels()." + x) for x in modelStatsModels().__all__]
            [mList.append("modelScikitLearn('" + x + "').runmodelEnsemble") for x in modelScikitLearn("Name").__all__]
            [mList.append("modelPyTorch()." + x) for x in modelPyTorch().__all__]

            sesonalRmGt60 = ['modelStatsModels().modelSARIMA', 'modelStatsModels().modelSARIMAX']
            if seasonal > 60:
                mList = list(set(mList) - set(sesonalRmGt60))

            start_time = datetime.datetime.now()
            if runType.lower() == "process":
                logging.info("Process Count :: {0}".format(processCount))
                with futures.ProcessPoolExecutor(max_workers=processCount) as furExec:
                    result = [furExec.map(self.submitModel, mList, itertools.repeat(seasonal),
                                          itertools.repeat(modelApproach), itertools.repeat(x_train),
                                          itertools.repeat(x_test), itertools.repeat(y_train), itertools.repeat(y_test),
                                          itertools.repeat(furDF.drop('Data', axis=1)))]
                    for mainRt in result:
                        for valRt in mainRt:
                            dataVal = dataVal + valRt
            else:
                logging.info("Thread Count :: {0}".format(threadCount if runType.lower() == "thread" else 1))
                with futures.ThreadPoolExecutor(
                        max_workers=threadCount if runType.lower() == "thread" else 1) as furExec:
                    result = [furExec.map(self.submitModel, mList, itertools.repeat(seasonal),
                                          itertools.repeat(modelApproach), itertools.repeat(x_train),
                                          itertools.repeat(x_test), itertools.repeat(y_train), itertools.repeat(y_test),
                                          itertools.repeat(furDF.drop('Data', axis=1)))]
                    for mainRt in result:
                        for valRt in mainRt:
                            dataVal = dataVal + valRt

            logging.info("Total Model Runtime {0} - {1} [ {2} ] ".format(start_time, datetime.datetime.now(),
                                                                         (datetime.datetime.now() - start_time)))

            return pd.DataFrame(dataVal, columns=columnsName)

        except Exception as e:
            print(e)

if __name__ == '__main__':
    print("Main")
