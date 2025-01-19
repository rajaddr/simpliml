from ..about import get_about

__date__ = get_about()['__date__']
__author__ = get_about()['__author__']
__version__ = get_about()['__version__']
__email__ = get_about()['__email__']
__description__ = get_about()['__description__']
__keywords__ = get_about()['__keywords__']
__url__ = get_about()['__url__']
__license__ = get_about()['__license__']
__status__ = get_about()['__status__']
del get_about

from .dataBuild import *
from .dataAnalysis import *
from .dataModeling import *


def generateTSData(dataDF, format='%Y-%m-%d', freq='D', periods=30):
    if dataDF.shape[1] == 2:
        return dataBuild().generateTSData(dataDF, format, freq, periods)
    else:
        return -1


def analysisData(dataDF):
    if dataDF.shape[1] >= 1:
        return dataAnalysis().analysisData(dataDF)
    else:
        return -1


def runModel(dataDF, furDF, seasonal=7, modelApproach='BEST', testSize=80):
    if (dataDF.shape[1] >= 1) & (furDF.shape[1] >= 1):
        return dataModeling().buildModel(dataDF, furDF, seasonal, modelApproach.upper(), testSize / 100, 'Single')
    else:
        print(dataDF.shape[1], furDF.shape[1])
        return -1


def runThreadModel(dataDF, furDF, seasonal=7, modelApproach='BEST', testSize=80):
    if (dataDF.shape[1] >= 1) & (furDF.shape[1] >= 1):
        return dataModeling().buildModel(dataDF, furDF, seasonal, modelApproach.upper(), testSize / 100, 'Thread')
    else:
        print(dataDF.shape[1], furDF.shape[1])
        return -1


def runProcessModel(dataDF, furDF, seasonal=7, modelApproach='BEST', testSize=80):
    if (dataDF.shape[1] >= 1) & (furDF.shape[1] >= 1):
        return dataModeling().buildModel(dataDF, furDF, seasonal, modelApproach.upper(), testSize / 100, 'Process')
    else:
        print(dataDF.shape[1], furDF.shape[1])
        return -1


def buildModel(dataDF, furDF, seasonal=7, modelApproach='BEST', testSize=80, runType='Single'):
    if (dataDF.shape[1] >= 1) & (furDF.shape[1] >= 1):
        return dataModeling().buildModel(dataDF, furDF, seasonal, modelApproach.upper(), testSize / 100, runType)
    else:
        print(dataDF.shape[1], furDF.shape[1])
        return -1


def modelResult(dataDF, mdlResult, modelApproach='Best'):
    if mdlResult.shape[0] > 0:
        return dataAnalysis().modelResult(dataDF, mdlResult, modelApproach)
    else:
        return -1
