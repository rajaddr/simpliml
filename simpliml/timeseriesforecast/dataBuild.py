import pandas as pd, numpy as np


class dataBuild:
    def applyDayCalc(self, df, col):
        try:
            df['Year'] = pd.to_datetime(df[col]).dt.year
            df['Month'] = pd.to_datetime(df[col]).dt.month
            df['Day'] = pd.to_datetime(df[col]).dt.day
            df['Dayofweek'] = pd.to_datetime(df[col]).dt.dayofweek
            df['Dayofyear'] = pd.to_datetime(df[col]).dt.dayofyear
            df['Week'] = pd.to_datetime(df[col]).dt.strftime('%U')  # .dt.week
            df['Quarter'] = pd.to_datetime(df[col]).dt.quarter
            df['Is_month_start'] = pd.to_datetime(df[col]).dt.is_month_start
            df['Is_month_end'] = pd.to_datetime(df[col]).dt.is_month_end
            df['Is_quarter_start'] = pd.to_datetime(df[col]).dt.is_quarter_start
            df['Is_quarter_end'] = pd.to_datetime(df[col]).dt.is_quarter_end
            df['Is_year_start'] = pd.to_datetime(df[col]).dt.is_year_start
            df['Is_year_end'] = pd.to_datetime(df[col]).dt.is_year_end
            df['Is_weekend'] = np.where(df['Dayofweek'].isin([5, 6]), 1, 0)
            df['Is_weekday'] = np.where(df['Dayofweek'].isin([0, 1, 2, 3, 4]), 1, 0)
            df['Days_in_month'] = pd.to_datetime(df[col]).dt.days_in_month
            df = pd.get_dummies(df,
                                columns=['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear', 'Week', 'Quarter',
                                         'Is_weekend',
                                         'Is_weekday', 'Days_in_month'])
            df.replace({False: 0, True: 1}, inplace=True)
            return df
        except Exception as e:
            print(e)

    def generateTSData(self, dataDF, format, freq, periods):
        try:
            dataDF.columns = ['Date', 'Data']
            dataDF['Date'] = pd.to_datetime(dataDF['Date'], format=format, errors='ignore')
            dataDF['Date'] = dataDF['Date'].dt.strftime(format)
            minDate = min(dataDF.Date)
            maxDate = max(dataDF.Date)
            dataSeq = pd.DataFrame(pd.date_range(minDate, maxDate, freq=freq), columns=['Date'])
            maxDateNew = max(
                pd.DataFrame(pd.date_range(maxDate, periods=2, freq=freq), columns=['Date']).drop(0).reset_index(
                    drop=True)['Date'])
            dataFur = pd.DataFrame(pd.date_range(maxDateNew, periods=periods, freq=freq), columns=['Date']).reset_index(
                drop=True)
            dataSeq['Date'] = dataSeq['Date'].dt.strftime(format)
            dataFur['Date'] = dataFur['Date'].dt.strftime(format)
            dataSet = pd.merge(dataDF, dataSeq, how="right", on=["Date"]).reset_index(drop=True)
            rtDF = self.applyDayCalc(
                pd.concat([dataSet, dataFur], axis=0).reset_index(drop=True).drop_duplicates("Date", keep='first'),
                'Date')
            return rtDF[rtDF['Date'] <= maxDate], rtDF[rtDF['Date'] > maxDate]
        except Exception as e:
            print(e)
