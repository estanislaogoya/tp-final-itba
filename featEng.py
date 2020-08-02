import pandas as pd
import numpy as np
QUOTE_DF = []

# As per Stock price predictions input variables for backpropagation
"""
- current stock price
− the absolute variation of the price in relation to previous day.
− direction of variation,
− direction of variation from two days previously,
− major variations in relation to the previous day
− the prices of the last 10 days (for Backpropagation)
"""

LOOK_WINDOW = -365

def putFuturePriceAndDeleteEmptyRows(df):
    df['future_price'] = df.loc[:, 'Close'].shift(LOOK_WINDOW)

    #Drop the n amount of rows, as they are empty
    df.drop(df.tail(abs(LOOK_WINDOW)).index,inplace=True)
    return df


def pricePredictionFeatureEng40(df):
    #assuming we recieve a 2 column df, date and price
    df.sort_values(by='Date',ascending=True)
    #df = df.rename(columns={1: 'Close'})
    df['1d_abs'] = df['Close'].diff(1)
    df['1d_dir'] = np.where((df['1d_abs'] <= 0), 0, 1)
    df['2d_abs'] = df['Close'].diff(2)
    df['1d_cls'] = df['Close'].shift(1)
    df['2d_cls'] = df['Close'].shift(2)
    df['3d_cls'] = df['Close'].shift(3)
    df['4d_cls'] = df['Close'].shift(4)
    df['5d_cls'] = df['Close'].shift(5)
    df['6d_cls'] = df['Close'].shift(6)
    df['7d_cls'] = df['Close'].shift(7)
    df['8d_cls'] = df['Close'].shift(8)
    df['9d_cls'] = df['Close'].shift(9)
    df['10d_cls'] = df['Close'].shift(10)
    #moving average indicators
    df['50d_ma'] = df['Close'].rolling(window=50).mean()
    df['100d_ma'] = df['Close'].rolling(window=100).mean()
    df['200d_ma'] = df['Close'].rolling(window=200).mean()
    #df['Volume'] = df['Volume'] / 100
    df = df.iloc[200:]
    return df

class StockDataFrame():
    name = ""
    data = []

    def __init__(self, df, name):
        self.name = name
        self.data = df
        return None

class DataframHandler():
#Breakdowns appended DataFrame and returns each object individually with FE applied to it
    objects = []
    fe_objects = []

    def df_breakdown(self, df_group):
        for _, group in df_group.groupby('asset'):
            self.objects.append(StockDataFrame(group.drop(['asset'], axis=1), group['asset'].iloc[0]))
        return self.objects

    def __init__(self, df, fe_method=None):
        for n in self.df_breakdown(df):
            if fe_method is None:
                fe_output = pricePredictionFeatureEng40(n.data)
            else:
                fe_output = fe_method(n.data)
            fe_output = putFuturePriceAndDeleteEmptyRows(fe_output)
            self.fe_objects.append(StockDataFrame(fe_output, n.name))
        return None
