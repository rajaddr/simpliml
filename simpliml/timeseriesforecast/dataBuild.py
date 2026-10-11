import pandas as pd
import numpy as np


class dataBuild:
    def applyDayCalc(self, df, col):
        try:
            # Convert datetime only once and reuse for all operations
            dt_col = pd.to_datetime(df[col])
            
            # Extract datetime components efficiently in one pass
            df['Year'] = dt_col.dt.year
            df['Month'] = dt_col.dt.month
            df['Day'] = dt_col.dt.day
            df['Dayofweek'] = dt_col.dt.dayofweek
            df['Dayofyear'] = dt_col.dt.dayofyear
            df['Week'] = dt_col.dt.strftime('%U')
            df['Quarter'] = dt_col.dt.quarter
            df['Is_month_start'] = dt_col.dt.is_month_start
            df['Is_month_end'] = dt_col.dt.is_month_end
            df['Is_quarter_start'] = dt_col.dt.is_quarter_start
            df['Is_quarter_end'] = dt_col.dt.is_quarter_end
            df['Is_year_start'] = dt_col.dt.is_year_start
            df['Is_year_end'] = dt_col.dt.is_year_end
            
            # Use vectorized operations for is_weekend/is_weekday
            is_weekend_mask = df['Dayofweek'].isin([5, 6])
            df['Is_weekend'] = is_weekend_mask.astype(int)
            df['Is_weekday'] = (~is_weekend_mask).astype(int)
            
            df['Days_in_month'] = dt_col.dt.days_in_month
            
            # Use get_dummies with drop_first for categorical encoding
            df = pd.get_dummies(
                df,
                columns=['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear', 'Week', 'Quarter',
                         'Is_weekend', 'Is_weekday', 'Days_in_month'],
                drop_first=False
            )
            
            # Replace boolean values using astype instead of replace
            bool_cols = df.select_dtypes(include=['bool']).columns
            df[bool_cols] = df[bool_cols].astype(int)
            
            return df
        except Exception as e:
            print(e)

    def generateTSData(self, dataDF, format, freq, periods):
        try:
            dataDF.columns = ['Date', 'Data']
            
            # Convert and format dates once
            dataDF['Date'] = pd.to_datetime(dataDF['Date'], format=format, errors='ignore')
            minDate = dataDF['Date'].min()
            maxDate = dataDF['Date'].max()
            
            # Format dates once for string comparison
            dataDF['Date'] = dataDF['Date'].dt.strftime(format)
            
            # Generate date sequences more efficiently
            dataSeq = pd.DataFrame(pd.date_range(minDate, maxDate, freq=freq), columns=['Date'])
            
            # Calculate next period date more efficiently
            next_date = pd.date_range(maxDate, periods=2, freq=freq)[1]
            dataFur = pd.DataFrame(pd.date_range(next_date, periods=periods, freq=freq), columns=['Date'])
            
            # Format dates for merging
            dataSeq['Date'] = dataSeq['Date'].dt.strftime(format)
            dataFur['Date'] = dataFur['Date'].dt.strftime(format)
            
            # Merge and deduplicate efficiently
            dataSet = pd.merge(dataDF, dataSeq, how="right", on=["Date"]).reset_index(drop=True)
            combined = pd.concat([dataSet, dataFur], axis=0, ignore_index=True).drop_duplicates("Date", keep='first')
            
            # Apply day calculations
            rtDF = self.applyDayCalc(combined, 'Date')
            
            # Filter efficiently
            return rtDF[rtDF['Date'] <= maxDate.strftime(format)], rtDF[rtDF['Date'] > maxDate.strftime(format)]
        except Exception as e:
            print(e)
