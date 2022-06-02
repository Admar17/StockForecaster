# Raw Package
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters

#Data Source
import yfinance as yf

#Data viz
import plotly.graph_objs as go



class stock:

    def __init__(self, ticker = None):
        self.ticker = ticker

    def getDailyStockInfo(self, ticker = None, period = '5y'):
        self.ticker = ticker
        print(ticker)


        if len([self.ticker]) == 1:
            
            self.daily_data = yf.download(tickers=self.ticker, period='5y', interval='1d')
        else:
            print('Error: Only try on stock ticker')
        
        return self.daily_data

    def plot_close_stock(self, data = None):
        # data = self.getDailyStockInfo(self.ticker)
        data = data
        self.data = data 

        data['Close'].plot(title = 'Daily Closing Price of\n' + str(self.ticker))

    def getETSPlot(self):
        data = self.data
        result_add = seasonal_decompose(data.Close,
model='additive', extrapolate_trend='freq', freq=365)
        plt.rcParams.update({'figure.figsize': (20, 10)})
        result_add.plot().suptitle('', fontsize=15)
        plt.show()

        return 

    def trim_data(self, data = None, month = None, year = None, day = 1):
        # self.trimmed_data = data
        self.month = month
        self.year = year
        self.day = day

        self.trimmed_data = data[data.index >= str(year)+'-'+str(month)+'-'+str(day)]

        return self.trimmed_data



