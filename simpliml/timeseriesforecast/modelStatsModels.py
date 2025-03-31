from .tsfCommon import *


class modelStatsModels:
    __all__ = ['modelHoltWinters', 'modelHolt', 'modelSimpleExpSmoothing', 'modelAutoRegression', 'modelMovingAverage',
               'modelARMA', 'modelARIMA', 'modelSARIMA', 'modelSARIMAX', 'modelVARMAX', 'modelVARMA']  # 'modelVAR' , 'modelAutoARIMA'

    def sarimaxSesOrder(self, y, seasonal):
        p = d = q = range(0, 2);
        c3 = [];
        c4 = []
        for param in list(itertools.product(p, d, q)):
            for param_seasonal in [(x[0], x[1], x[2], seasonal) for x in list(itertools.product(p, d, q))]:
                try:
                    results = SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False)
                    if np.isnan(results.aic) == False:
                        c3.append(results.aic)
                        c4.append([results.aic, param, param_seasonal])
                except:
                    continue
        return list(c4[np.argmin(c3)][1]), list(c4[np.argmin(c3)][2])

    def arimaSesOrder(self, y, seasonal, p=range(0, 2), d=range(0, 2), q=range(0, 2)):
        c3 = [];
        c4 = []
        for param in list(itertools.product(p, d, q)):
            try:
                results = ARIMA(y, order=param, enforce_stationarity=False, enforce_invertibility=False).fit()
                if np.isnan(results.aic) == False:
                    c3.append(results.aic)
                    c4.append([results.aic, param])
            except:
                continue
        return list(c4[np.argmin(c3)][1])

    def modelHoltWinters(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            best_RMSE = np.inf
            best = []
            for t in ['add', 'mul', None]:
                for d in [True, False]:
                    for s in ['add', 'mul', None]:
                        for r in [True, False]:
                            try:
                                if (t == None):
                                    model = ExponentialSmoothing(y_train, trend=t, damped=d, seasonal=s).fit(
                                        optimized=True, remove_bias=r)
                                else:
                                    model = ExponentialSmoothing(y_train, trend=t, damped=d, seasonal=s).fit(
                                        optimized=True, remove_bias=r)
                                rmse = np.sqrt(mean_squared_error(y_test, model.forecast(len(y_test))))

                                if rmse <= best_RMSE:
                                    best_RMSE = rmse
                                    best_t, best_d, best_s, best_r = t, d, s, r
                            except Exception:
                                continue

            if (best_t == None):
                modelTest = ExponentialSmoothing(y_train, trend=best_t, seasonal=best_s).fit(optimized=True,
                                                                                             remove_bias=best_r).forecast(
                    len(y_test))
                modelFur = ExponentialSmoothing(pd.concat([y_train, y_test], axis=0), trend=best_t,
                                                seasonal=best_s).fit(optimized=True, remove_bias=best_r).forecast(
                    len(furDF))
            else:
                modelTest = ExponentialSmoothing(y_train, trend=best_t, damped=best_d, seasonal=best_s).fit(
                    optimized=True, remove_bias=best_r).forecast(len(y_test))
                modelFur = ExponentialSmoothing(pd.concat([y_train, y_test], axis=0), trend=best_t, damped=best_d,
                                                seasonal=best_s).fit(optimized=True, remove_bias=best_r).forecast(
                    len(furDF))

            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred[0])
            rtVal.append(
                ["statsmodels.tsa.holtwinters", "ExponentialSmoothing", True, modelSt[0], modelSt[1], modelSt[2],
                 modelSt[3], modelSt[4], modelSt[5], modelSt[6], modelSt[7], modelSt[8], modelSt[9],
                 pred[['Date', 0]].values.tolist(), furData[['Date', 0]].values.tolist()])
        except Exception as e:
            rtVal.append(
                ["statsmodels.tsa.holtwinters", "ExponentialSmoothing", False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal

    def modelHolt(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            modelTest = Holt(y_train).fit(optimized=True, remove_bias=True).forecast(len(y_test))
            modelFur = Holt(pd.concat([y_train, y_test], axis=0)).fit(optimized=True, remove_bias=True).forecast(
                len(furDF))
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred[0])
            rtVal.append(["statsmodels.tsa.holtwinters", "Holt", True, modelSt[0], modelSt[1], modelSt[2], modelSt[3],
                          modelSt[4], modelSt[5], modelSt[6], modelSt[7], modelSt[8], modelSt[9],
                          pred[['Date', 0]].values.tolist(), furData[['Date', 0]].values.tolist()])
        except Exception as e:
            rtVal.append(["statsmodels.tsa.holtwinters", "Holt", False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal

    def modelSimpleExpSmoothing(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            modelTest = SimpleExpSmoothing(y_train).fit(optimized=True, remove_bias=True).forecast(len(y_test))
            modelFur = SimpleExpSmoothing(pd.concat([y_train, y_test], axis=0)).fit(optimized=True,
                                                                                    remove_bias=True).forecast(
                len(furDF))
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred[0])
            rtVal.append(["statsmodels.tsa.holtwinters", "SimpleExpSmoothing", True, modelSt[0], modelSt[1], modelSt[2],
                          modelSt[3], modelSt[4], modelSt[5], modelSt[6], modelSt[7], modelSt[8], modelSt[9],
                          pred[['Date', 0]].values.tolist(), furData[['Date', 0]].values.tolist()])
        except Exception as e:
            rtVal.append(
                ["statsmodels.tsa.holtwinters", "SimpleExpSmoothing", False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal

    def modelAutoRegression(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            modelTest = AutoReg(y_train, lags=1).fit().forecast(len(y_test))
            modelFur = AutoReg(pd.concat([y_train, y_test], axis=0), lags=1).fit().forecast(len(furDF))
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred[0])
            rtVal.append(["statsmodels.tsa.ar_model", "Autoregression (AR)", True, modelSt[0], modelSt[1], modelSt[2],
                          modelSt[3], modelSt[4], modelSt[5], modelSt[6], modelSt[7], modelSt[8], modelSt[9],
                          pred[['Date', 0]].values.tolist(), furData[['Date', 0]].values.tolist()])
        except Exception as e:
            rtVal.append(
                ["statsmodels.tsa.ar_model", "Autoregression (AR)", False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal

    def modelMovingAverage(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            modelTest = ARIMA(y_train, order=self.arimaSesOrder(y_train, seasonal, p=[0], d=[0])).fit().forecast(
                len(y_test))
            modelFur = ARIMA(pd.concat([y_train, y_test], axis=0),
                             order=self.arimaSesOrder(pd.concat([y_train, y_test], axis=0), seasonal, p=[0],
                                                      d=[0])).fit().forecast(len(furDF))
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred['predicted_mean'])
            rtVal.append(
                ["statsmodels.tsa.arima.model", "Moving Average (MA)", True, modelSt[0], modelSt[1], modelSt[2],
                 modelSt[3], modelSt[4], modelSt[5], modelSt[6], modelSt[7], modelSt[8], modelSt[9],
                 pred[['Date', 'predicted_mean']].values.tolist(), furData[['Date', 'predicted_mean']].values.tolist()])
        except Exception as e:
            rtVal.append(
                ["statsmodels.tsa.arima.model", "Moving Average (MA)", False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal

    def modelARMA(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            modelTest = ARIMA(y_train, order=self.arimaSesOrder(y_train, seasonal, d=[0])).fit().forecast(len(y_test))
            modelFur = ARIMA(pd.concat([y_train, y_test], axis=0),
                             order=self.arimaSesOrder(pd.concat([y_train, y_test], axis=0), seasonal,
                                                      d=[0])).fit().forecast(len(furDF))
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred['predicted_mean'])
            rtVal.append(
                ["statsmodels.tsa.arima.model", "Autoregressive Moving Average (ARMA)", True, modelSt[0], modelSt[1],
                 modelSt[2], modelSt[3], modelSt[4], modelSt[5], modelSt[6], modelSt[7], modelSt[8], modelSt[9],
                 pred[['Date', 'predicted_mean']].values.tolist(), furData[['Date', 'predicted_mean']].values.tolist()])
        except Exception as e:
            rtVal.append(
                ["statsmodels.tsa.arima.model", "Autoregressive Moving Average (ARMA)", False, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, [], []])
            print(e)

        return rtVal

    def modelARIMA(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            modelTest = ARIMA(y_train, order=self.arimaSesOrder(y_train, seasonal)).fit().forecast(len(y_test))
            modelFur = ARIMA(pd.concat([y_train, y_test], axis=0),
                             order=self.arimaSesOrder(pd.concat([y_train, y_test], axis=0), seasonal)).fit().forecast(
                len(furDF))
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred['predicted_mean'])
            rtVal.append(
                ["statsmodels.tsa.arima.model", "Autoregressive Integrated Moving Average (ARIMA)", True, modelSt[0],
                 modelSt[1], modelSt[2], modelSt[3], modelSt[4], modelSt[5], modelSt[6], modelSt[7], modelSt[8],
                 modelSt[9], pred[['Date', 'predicted_mean']].values.tolist(),
                 furData[['Date', 'predicted_mean']].values.tolist()])
        except Exception as e:
            rtVal.append(
                ["statsmodels.tsa.arima.model", "Autoregressive Integrated Moving Average (ARIMA)", False, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal

    def modelSARIMA(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            order, senl = self.sarimaxSesOrder(y_train, seasonal)
            modelTest = SARIMAX(y_train, order=order, seasonal_order=senl, enforce_stationarity=False,
                                enforce_invertibility=False).fit(disp=False).forecast(len(y_test))
            order, senl = self.sarimaxSesOrder(pd.concat([y_train, y_test], axis=0), seasonal)
            modelFur = SARIMAX(pd.concat([y_train, y_test], axis=0), order=order, seasonal_order=senl,
                               enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(
                len(furDF))
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred['predicted_mean'])
            rtVal.append(
                ["statsmodels.tsa.statespace.sarimax", "Seasonal Autoregressive Integrated Moving-Average (SARIMA)",
                 True, modelSt[0], modelSt[1], modelSt[2], modelSt[3], modelSt[4], modelSt[5], modelSt[6], modelSt[7],
                 modelSt[8], modelSt[9], pred[['Date', 'predicted_mean']].values.tolist(),
                 furData[['Date', 'predicted_mean']].values.tolist()])
        except Exception as e:
            rtVal.append(
                ["statsmodels.tsa.statespace.sarimax", "Seasonal Autoregressive Integrated Moving-Average (SARIMA)",
                 False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal

    def modelSARIMAX(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            y_train1 = pd.concat([y_train, y_test], axis=0)
            order, senl = self.sarimaxSesOrder(y_train, seasonal)
            modelTest = SARIMAX(y_train, exog=np.random.randint(min(y_train), max(y_train), len(y_train)), order=order,
                                seasonal_order=senl).fit(disp=False).forecast(len(y_test),
                                                                              exog=np.random.randint(min(y_test),
                                                                                                     max(y_test),
                                                                                                     len(y_test)))
            order, senl = self.sarimaxSesOrder(pd.concat([y_train, y_test], axis=0), seasonal)
            modelFur = SARIMAX(y_train1, exog=np.random.randint(min(y_train1), max(y_train1), len(y_train1)),
                               order=order, seasonal_order=senl).fit(disp=False).forecast(len(furDF),
                                                                                          exog=np.random.randint(
                                                                                              min(y_train1),
                                                                                              max(y_train1),
                                                                                              len(furDF)))
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred['predicted_mean'])
            rtVal.append(["statsmodels.tsa.statespace.sarimax",
                          "Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)", True,
                          modelSt[0], modelSt[1], modelSt[2], modelSt[3], modelSt[4], modelSt[5], modelSt[6],
                          modelSt[7], modelSt[8], modelSt[9], pred[['Date', 'predicted_mean']].values.tolist(),
                          furData[['Date', 'predicted_mean']].values.tolist()])
        except Exception as e:
            rtVal.append(["statsmodels.tsa.statespace.sarimax",
                          "Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)",
                          False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal

    def modelVAR(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            modelTest = VAR(y_train.reset_index(drop=True).reset_index()).fit().forecast(len(y_test), steps=1)
            modelFur = VAR(pd.concat([y_train, y_test], axis=0)).fit().forecast(len(furDF), steps=1)
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred['predicted_mean'])
            rtVal.append(
                ["statsmodels.tsa.vector_ar.var_model", "Vector Autoregression (VAR)", True, modelSt[0], modelSt[1],
                 modelSt[2], modelSt[3], modelSt[4], modelSt[5], modelSt[6], modelSt[7], modelSt[8], modelSt[9],
                 pred[['Date', 'predicted_mean']].values.tolist(), furData[['Date', 'predicted_mean']].values.tolist()])
        except Exception as e:
            rtVal.append(
                ["statsmodels.tsa.vector_ar.var_model", "Vector Autoregression (VAR)", False, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, [], []])
            print(e)

        return rtVal

    def modelVARMA(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            modelTest = VARMAX(y_train.reset_index(drop=True).reset_index(), order=(1, 1)).fit(disp=False).forecast(
                len(y_test))
            modelFur = VARMAX(pd.concat([y_train, y_test], axis=0).reset_index(drop=True).reset_index(),
                              order=(1, 1)).fit(disp=False).forecast(len(furDF))
            modelTest.columns = ['index', 'predicted_mean']
            modelFur.columns = ['index', 'predicted_mean']
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred['predicted_mean'])
            rtVal.append(
                ["statsmodels.tsa.statespace.varmax", "Vector Autoregression Moving-Average (VARMA)", True, modelSt[0],
                 modelSt[1], modelSt[2], modelSt[3], modelSt[4], modelSt[5], modelSt[6], modelSt[7], modelSt[8],
                 modelSt[9], pred[['Date', 'predicted_mean']].values.tolist(),
                 furData[['Date', 'predicted_mean']].values.tolist()])
        except Exception as e:
            rtVal.append(
                ["statsmodels.tsa.statespace.varmax", "Vector Autoregression Moving-Average (VARMA)", False, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal

    def modelVARMAX(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            y_train1 = pd.concat([y_train, y_test], axis=0)
            modelTest = VARMAX(y_train.reset_index(drop=True).reset_index(),
                               exog=np.random.randint(min(y_train), max(y_train), len(y_train)), order=(1, 1)).fit(
                disp=False).forecast(len(y_test), exog=np.random.randint(min(y_test), max(y_test), len(y_test)))
            modelFur = VARMAX(y_train1.reset_index(drop=True).reset_index(),
                              exog=np.random.randint(min(y_train1), max(y_train1), len(y_train1)), order=(1, 1)).fit(
                disp=False).forecast(len(furDF), exog=np.random.randint(min(y_train1), max(y_train1), len(furDF)))
            modelTest.columns = ['index', 'predicted_mean']
            modelFur.columns = ['index', 'predicted_mean']
            pred = pd.concat([y_test.reset_index(), modelTest.reset_index(drop=True)], axis=1)
            furData = pd.concat([furDF.reset_index(), modelFur.reset_index(drop=True)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred['predicted_mean'])
            rtVal.append(["statsmodels.tsa.statespace.varmax",
                          "Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)", True, modelSt[0],
                          modelSt[1], modelSt[2], modelSt[3], modelSt[4], modelSt[5], modelSt[6], modelSt[7],
                          modelSt[8], modelSt[9], pred[['Date', 'predicted_mean']].values.tolist(),
                          furData[['Date', 'predicted_mean']].values.tolist()])
        except Exception as e:
            rtVal.append(["statsmodels.tsa.statespace.varmax",
                          "Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)", False, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal

    def modelAutoARIMA(self, seasonal, x_train, x_test, y_train, y_test, furDF, modelApproach):
        rtVal = []
        try:
            modelTest = auto_arima(y_train, seasonal=True, trace=False, error_action='ignore',
                                   suppress_warnings=True).fit(y_train).predict(len(y_test))
            modelFur = auto_arima(pd.concat([y_train, y_test], axis=0), seasonal=True, trace=False,
                                  error_action='ignore', suppress_warnings=True).fit(
                pd.concat([y_train, y_test], axis=0)).predict(len(furDF))
            pred = pd.concat([y_test.reset_index(), pd.DataFrame(modelTest)], axis=1)
            furData = pd.concat([furDF.reset_index(), pd.DataFrame(modelFur)], axis=1)
            modelSt = tsfCommon().modelEval(pred['Data'], pred[0])
            rtVal.append(
                ["pmdarima.arima", "Auto ARIMA", True, modelSt[0], modelSt[1], modelSt[2], modelSt[3], modelSt[4],
                 modelSt[5], modelSt[6], modelSt[7], modelSt[8], modelSt[9], pred[['Date', 0]].values.tolist(),
                 furData[['Date', 0]].values.tolist()])
        except Exception as e:
            rtVal.append(["pmdarima.arima", "Auto ARIMA", False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], []])
            print(e)

        return rtVal
