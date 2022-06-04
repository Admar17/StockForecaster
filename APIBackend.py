# Raw Package
from unittest import result
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from matplotlib import pyplot as plt

from dateutil.relativedelta import relativedelta, MO
import altair as alt
from datetime import datetime




#Data Source
import yfinance as yf

#Data viz
import plotly.graph_objs as go
import seaborn as sns



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
        try:
            self.trimmed_data = data[data.index >= str(year)+'-'+str(month)+'-'+str(day)]
        except:
            self.trimmed_data = data
        self.X = self.trimmed_data.index
        self.y = self.trimmed_data.Close    

        return self.trimmed_data

    def plotDiagnostics(self):
        diff_stock = self.trimmed_data.dropna()
        fig = plt.figure(figsize=(20,12))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(diff_stock.Close,lags=24,ax=ax1)
        ax2 = fig.add_subplot(222)
        fig = sm.graphics.tsa.plot_pacf(diff_stock.Close,lags=24,ax=ax2)

    def ARMA(self,ar,ma):
        model = sm.tsa.ARMA(self.trimmed_data.Close, order = (ar, ma))
        results = model.fit()
        self.results = results
        print(results.summary())
        self.mergeForecastActuals()


    def plotARMAFit(self):
        results = self.results
        # Define figure style, plot package and default figure size
        sns.set_style("darkgrid")
        pd.plotting.register_matplotlib_converters()
        
        # Default figure size
        sns.mpl.rc("figure", figsize=(16, 6))
        
        # Use plot_predict and visualize forecasts
        figure = results.plot_predict(dynamic=False, plot_insample=True)


        return

    def errorTerms(self):
        results = self.results
        y = self.y
        print(f'MAE: {sum(np.abs(results.fittedvalues-y))/len(y)}')  ## Error Term Mean Abosult Error; MAE Measure On Average we are xx.xx dollars of from the actual
        print(f'MAPE: {sum(np.abs(results.fittedvalues-y)/y)/len(y)}' ) ## Error Term Mean Abosulte Percent Error; MAPE Measure on average what is the percent difference from actual

    def mergeForecastActuals(self):

        results = self.results
        y = self.y
        df = self.trimmed_data

        df=pd.DataFrame(y)
        df.index.name = ''
        df['Result'] = 'Actual'

        days_to_forecast = 365 ## You Change This


        try:
            forecasted_results =results.forecast(steps = days_to_forecast)[0]
        except:
            forecasted_results =results.forecast(steps = days_to_forecast)

        start_date = y.index[-1] + relativedelta(days=1)

        forecast_index = pd.date_range(start = start_date, periods=len(forecasted_results), freq = 'B')
        forecast_df = pd.DataFrame(data={'Close':forecasted_results})
        forecast_df.index = forecast_index
        forecast_df['Result'] = 'Forecasted'
        combined_df =pd.concat([df,forecast_df], axis = 0)
        combined_df['Close'] = np.round(combined_df['Close'],2)

        self.combined_df = combined_df
        return 

    def forecastPlot(self):
        self.errorTerms()

        combined_df = self.combined_df

        ## Second Chart

        ### https://altair-viz.github.io/gallery/multiline_tooltip.html

        # Create a selection that chooses the nearest point & selects based on x-value
        nearest = alt.selection(type='single', nearest=True, on='mouseover',
                                fields=['index','Close'], empty='none')

        # The basic line
        line = alt.Chart(combined_df.reset_index()).mark_line(interpolate='basis').encode(
            x =  'index:T',
            y = 'Close:Q',
            color = 'Result'
            # tooltip = ['index','Close','Result']
        )

        selectors = alt.Chart(combined_df.reset_index()).mark_point().encode(
            x='index:T',
            opacity=alt.value(0),
        ).add_selection(
            nearest
        )

        # Draw points on the line, and highlight based on selection
        points = line.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

        # Draw text labels near the points, and highlight based on selection
        text = line.mark_text(align='left', dx=20, dy = -15, size = 20).encode(
            text=alt.condition(nearest, 'Close:Q', alt.value(' '))
        )

        text_index = line.mark_text(align='left', dx=5, dy=-40, size = 20).encode(
            text=alt.condition(nearest, 'index:T', alt.value(''))
        )

        rules = alt.Chart(combined_df.reset_index()).mark_rule(color='gray').encode(
            x='index:T',
        ).transform_filter(
            nearest
        )

        # Put the five layers into a chart and bind the data
        return alt.layer(
            line, selectors, points, text_index, text 
        ).properties(
            width=1200, height=600
        ).interactive()

    def ARIMA(self,ar,ma):
        model = sm.tsa.arima.ARIMA(self.trimmed_data.Close, order = (ar, 1,ma), seasonal_order = (1,1,1,52))
        results = model.fit()
        self.results = results
        print(results.summary())
        self.mergeForecastActuals()

    def saveDataFrame(self, save = False):
        combined_df = self.combined_df
        ticker = self.ticker
        if save:
            today_string =datetime.today().strftime('%d-%m-%Y')
            filename = today_string+'_'+ticker+'_Forecast.csv'
            combined_df.to_csv(filename)
            print(f'DateFrame was saved as: {filename}')
        else:
            print(f'No DataFrame was saved for {ticker}')


                    