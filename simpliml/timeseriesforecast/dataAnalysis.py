import logging, time
import pandas as pd, numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import jarque_bera as jb

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


class dataAnalysis:
    def __init__(self):
        self.graphWidth = 800
        self.graphHeight = 200
        self.opacityVal = 0.9
        self.strokeOpacityVal = 0.4

    def analysisData(self, dfPass):
        try:
            df = dfPass.copy()
            df['Data'] = df['Data'].fillna(df['Data'].mean())
            allData = df[df['Data'] != 0]['Data'].values
            logData = np.log(allData)
            adfAllRst = adfuller(allData)
            adfLogRst = adfuller(logData)
            is_norm = jb(logData)
            df['logData'] = np.log(df[df['Data'] != 0]['Data'])
            df['rolmean'] = pd.Series(df['Data']).rolling(window=12).mean()
            df['rolstd'] = pd.Series(df['Data']).rolling(window=12).std()

            newDF = df[['Date', 'Data']].set_index('Date')
            print(newDF.describe().transpose())
            print("p value : {0}, Series is {1}".format(adfAllRst[1].round(5),
                                                        "Stationary" if (adfAllRst[1] <= 0.05) & (
                                                                adfLogRst[0] >= 0) else "Non-Stationary"))
            print("p value : {0}, Series is {1}".format(is_norm[1].round(5),
                                                        "Normal" if (is_norm[1] > 0.05) else "Non-Normal"))

            mC1 = alt.Chart(df, title="Data Series With Rolling Mean").mark_line(point=alt.OverlayMarkDef()).encode(
                x='Date', y='Data',
                tooltip=['Date', 'Data'],
                opacity=alt.value(self.opacityVal),
                strokeOpacity=alt.value(self.strokeOpacityVal),
                color=alt.value("#01C0C0")
            )
            mA1 = alt.Chart(df).mark_trail(size=1, color='red').encode(
                x='Date', y='rolmean',
                size='Data'
            )
            mT1 = mC1.mark_text(align='left', baseline='top', dx=5).encode(text='Data:Q')

            (mC1 + mA1).properties(width=self.graphWidth, height=self.graphHeight).interactive().display()

            mC2 = alt.Chart(df, title="Sub Series").mark_line(point=alt.OverlayMarkDef()).encode(
                x='quarter(Date)',
                y='Data',
                column='year(Date)',
                color=alt.value("#01C0C0"),
                tooltip=['Date', 'Data']).configure_header(
                titleColor='black',
                titleFontSize=14,
                labelColor='blue',
                labelFontSize=14
            )
            (mC2).properties(width=80, height=self.graphHeight).interactive().display()

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=df, x=pd.DatetimeIndex(df['Date']).year, y='Data', ax=ax, boxprops=dict(alpha=.3))
            sns.swarmplot(data=df, x=pd.DatetimeIndex(df['Date']).year, y='Data')
            plt.show()

            mC3 = alt.Chart(df, title="Quarterly Trends").mark_line(point=alt.OverlayMarkDef()).encode(
                x='year(Date)',
                y='Data',
                column='quarter(Date)',
                color=alt.value("#01C0C0"),
                tooltip=['Date', 'Data']).configure_header(
                titleColor='black',
                titleFontSize=14,
                labelColor='blue',
                labelFontSize=14
            )
            (mC3).properties(width=200, height=self.graphHeight).interactive().display()

            mC4 = alt.Chart(df, title="Sum by each Quarter").mark_bar(point=alt.OverlayMarkDef()).encode(
                x='sum(Data)',
                y='year(Date):N',
                color=alt.Color('quarter(Date)', scale=alt.Scale(scheme='category10')),
                tooltip=["Date", "Data"])
            (mC4).properties(width=self.graphWidth, height=self.graphHeight).interactive().display()

            mC5 = alt.Chart(df, title="Sum by each Quarter").mark_bar(point=alt.OverlayMarkDef()).encode(
                x=alt.X('sum(Data)', stack='normalize'),
                y='year(Date):N',
                color=alt.Color('quarter(Date)', scale=alt.Scale(scheme='category10')),
                tooltip=["Date", "Data"])
            (mC5).properties(width=self.graphWidth, height=self.graphHeight).interactive().display()

            alt.Chart(df, title="White Noise").mark_line(point=alt.OverlayMarkDef()).encode(
                x='Date', y='Data',
                tooltip=[alt.Tooltip('Data'), alt.Tooltip('Date')],
                opacity=alt.value(self.opacityVal),
                strokeOpacity=alt.value(self.strokeOpacityVal),
                color=alt.value("#01C0C0")
            ).properties(width=self.graphWidth, height=self.graphHeight).add_selection(
                alt.selection_interval(bind='scales')).display()

            alt.Chart(df, title="Log Histogram").mark_bar(filled=True).encode(
                alt.X('logData', bin=False), alt.Y('count()'),
                opacity=alt.value(self.opacityVal),
                strokeOpacity=alt.value(self.strokeOpacityVal),
                tooltip=[alt.Tooltip('logData'), alt.Tooltip('count()')],
                color=alt.value("#01C0C0")
            ).properties(width=self.graphWidth, height=self.graphHeight).display()

            alt.Chart(df, title="Log Line Plot").mark_line(point=alt.OverlayMarkDef()).encode(
                x='Date', y='logData',
                opacity=alt.value(self.opacityVal),
                strokeOpacity=alt.value(self.strokeOpacityVal),
                tooltip=[alt.Tooltip('logData'), alt.Tooltip('Date')],
                color=alt.value("#01C0C0")
            ).properties(width=self.graphWidth, height=self.graphHeight).display()

            plt.rcParams.update({'figure.figsize': (20, 12)})
            seasonal_decompose(newDF, model='additive', period=1).plot().suptitle('nAdditive Decompose')
            newDF.plot(figsize=(12, 6), legend=True, label="Train", cmap='gray')
            newDF['Data'].rolling(4, center=False).mean().plot(legend=True, label="Rolling Mean 4Q")
            sm.graphics.tsa.plot_acf(newDF)
            sm.graphics.tsa.plot_pacf(newDF)
            plt.show()
            sns.distplot(newDF)

            plt.show()

        except Exception as e:
            print(e)

    def modelResult(self, dataDF, mdlResult, modelApproach):
        try:
            if modelApproach.upper() == 'BEST':
                bestData = mdlResult[mdlResult['MAPE'] == mdlResult['MAPE'][mdlResult['Status'] == True].min()]
            else:
                bestData = mdlResult[mdlResult['Model'] == modelApproach]
            dataTest = pd.DataFrame(bestData['Test'].values[0], columns=['Date', 'Data'])
            dataTest['Value'] = 'Test'
            datafur = pd.DataFrame(bestData['Predict'].values[0], columns=['Date', 'Data'])
            datafur['Value'] = 'Forecast'
            dataOrg = dataDF.reset_index()[['Date', 'Data']]
            dataOrg['Value'] = 'Data'
            source = pd.concat([dataTest, datafur, dataOrg], axis=0)

            nearest = alt.selection(type='single', nearest=True, on='mouseover',
                                    fields=['Date'], empty='none')

            line = alt.Chart(source).mark_line(interpolate='basis').encode(
                x='Date',
                y='Data:Q',
                color='Value:N'
            )

            selectors = alt.Chart(source).mark_point().encode(
                x='Date',
                opacity=alt.value(0),
            ).add_selection(
                nearest
            )
            points = line.mark_point().encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0))
            )

            text = line.mark_text(align='left', dx=5, dy=-5).encode(
                text=alt.condition(nearest, 'Data:Q', alt.value(' '))
            )

            text1 = line.mark_text(align='right', dx=-5, dy=5).encode(
                text=alt.condition(nearest, 'Date', alt.value(' '))
            )

            rules = alt.Chart(source).mark_rule(color='gray').encode(
                x='Date',
            ).transform_filter(
                nearest
            )
            dataValue = pd.concat([dataDF.reset_index()[['Date', 'Data']].set_index('Date'),
                                   pd.DataFrame(bestData['Test'].values[0], columns=['Date', 'Test']).set_index('Date'),
                                   pd.DataFrame(bestData['Predict'].values[0], columns=['Date', 'Forecast']).set_index(
                                       'Date')], axis=1).fillna(0).reset_index()
            dataSt = bestData[
                ['Model', 'MAE', 'MSE', 'RMSE', 'RMSLE', 'MAPE', 'R2', 'SMAPE', 'MFE', 'NMSE', 'THEIL_U_STATISTIC']]
            table = [dataSt.columns.values.tolist()] + dataSt.sort_values('MAPE').values.tolist()
            longest_cols = [
                (max([len(str(row[i])) for row in table]) + 2)
                for i in range(len(table[0]))
            ]

            try:
                alt.layer(
                    line, selectors, points, rules, text, text1
                ).properties(
                    width=self.graphWidth, height=self.graphHeight
                ).interactive().display()

                row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
                for row in table:
                    print(row_format.format(*row))
            except:
                pass

            return dataValue, dataSt

        except Exception as e:
            print(e)
            return -1
