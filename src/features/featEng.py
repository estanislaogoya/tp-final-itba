import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

    df['future_price'] = df.loc[:, 'Close'].shift(LOOK_WINDOW)

    #Drop the n amount of rows, as they are empty
    df.drop(df.tail(abs(LOOK_WINDOW)).index,inplace=True)
    return df


def pricePredictionFeatureEng_B(df):
    #assuming we recieve a 2 column df, date and price
    df.sort_values(by='Date',ascending=True)
    #df = df.rename(columns={1: 'Close'})
    df['1d_abs'] = df['Close'].diff(1)
    df['1d_dir'] = np.where((df['1d_abs'] <= 0), 0, 1)
    df['2d_abs'] = df['Close'].diff(2)
    df['50d_cls'] = df['Close'].shift(50)
    df['100d_cls'] = df['Close'].shift(100)
    df['150d_cls'] = df['Close'].shift(150)
    df['200d_cls'] = df['Close'].shift(200)
    df['250d_cls'] = df['Close'].shift(250)
    df['300d_cls'] = df['Close'].shift(300)
    df['50d_dir'] = np.where((df['Close'] >= (df['50d_cls'])), 1, 0)
    df['100d_dir'] = np.where((df['Close'] >= (df['100d_cls'])), 1, 0)
    df['150d_dir'] = np.where((df['Close'] >= (df['150d_cls'])), 1, 0)
    df['200d_dir'] = np.where((df['Close'] >= (df['200d_cls'])), 1, 0)
    df['250d_dir'] = np.where((df['Close'] >= (df['250d_cls'])), 1, 0)
    df['300d_dir'] = np.where((df['Close'] >= (df['300d_cls'])), 1, 0)
    #moving average indicators
    df['50d_ma'] = df['Close'].rolling(window=50).mean()
    df['100d_ma'] = df['Close'].rolling(window=100).mean()
    df['200d_ma'] = df['Close'].rolling(window=200).mean()
    df['300d_ma'] = df['Close'].rolling(window=300).mean()
    #df['Volume'] = df['Volume'] / 100
    df = df.iloc[300:]

    df['future_price'] = df.loc[:, 'Close'].shift(LOOK_WINDOW)

    #Drop the n amount of rows, as they are empty
    df.drop(df.tail(abs(LOOK_WINDOW)).index,inplace=True)
    return df

def splittingForTraining(df):
    target = df.pop('future_price')
    return train_test_split(df,
                            target,
                            test_size=0.33,
                            random_state=42)
