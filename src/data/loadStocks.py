import yfinance as yf
import pandas as pd
import os
import numpy as np

DF_SET = []
DF_OBJ = []
DIR_STOCKS = 'data/stocks/'

#adding text

#Check if stocks already exist on disk to reduce API calls
def stock_dir_exists():
    if not os.path.exists(DIR_STOCKS):
        os.makedirs(DIR_STOCKS)

#If file does not exist, then fetch using yahoo finance
def fetch_quotes_yf(ticker):
    price_df = pd.DataFrame(yf.Ticker(str(ticker)).history(period="10y"), columns=['Open','High','Low','Close','Volume'])
    price_df.to_csv(DIR_STOCKS + str(ticker))
    return price_df

#test
# display is the way how the data will be bundled.
# display = {"APPEND", "GRID", "OBJECTS"}
def load_tickers_df(ticker_names, display = "objects"):
    stock_dir_exists()
    for i in ticker_names:
        ticker_filename = DIR_STOCKS + str(i) + '.csv'
        if os.path.isfile(ticker_filename) == False:
            df = fetch_quotes_yf(i)
        else:
            df = pd.read_csv(ticker_filename)

        if display == "grid":
            #pass
            df.drop(df.columns.difference(['Close']), 1, inplace=True)
        else:
            df['asset'] = i
        DF_SET.append(df)
    if display == "grid":
        return pd.concat(DF_SET, axis = 1)
    elif display == "append":
        return pd.concat(DF_SET)
    else:
        return DF_SET
