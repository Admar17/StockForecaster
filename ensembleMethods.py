import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta, MO
import altair as alt
from datetime import datetime
import APIBackend as back
import matplotlib.pyplot as plt



class baseModel():
    
    def __init__(self, ticker = None,trim_month = None,  trim_year = None):
        self.ticker = ticker
        self.trim_month = trim_month
        self.trim_year = trim_year
        
        self.getData(ticker=ticker, month=trim_month, year=trim_year)
        
        self.x = self.trimmed_data.index
        self.y = self.trimmed_data.Close      
        
        self.train_test_split()
        

        
    def getData(self, ticker = None, month = None, year = None):
        ticker = self.ticker 
        trim_month = self.trim_month
        trim_year = self.trim_year
        data = back.stock().getDailyStockInfo(ticker = ticker)
        trimmed_data = back.stock().trim_data(data=data, month = trim_month, year = trim_year)
        
        self.trimmed_data = trimmed_data
    
        
        
    def naiveForecast(self, train_test = False, save_model = True):
        
        self.model_name = 'Naive'
        if train_test == True:
            y = self.y_train 
        else:
            y = self.y
            
        self.y = y
            
        
        period = self.getPeriods()
        
        
            
        start_date = y.index[-1] + relativedelta(days=1)
        self.forecasted_results = np.repeat(y[-1], period)       
        self.NaiveDF = self.mergeDataFrame()
        
        if train_test == True:
            self.getEvaluation()
            self.getTrainForecastPlot()
        
        if save_model == True:
            self.saved_forecasts_.append(self.combined_df)
        
    
    def driftForecast(self,train_test = False, save_model = True):
        
        self.model_name = 'Drift'
        if train_test == True:
            y = self.y_train 
        else:
            y = self.y
            
        self.y = y
        
        self.forecasted_results = []
       
        avg_drift = (y[-1]+y[1])/len(y)
        period = self.getPeriods()
        forecast_value= y[-1]

        for i in range(period):
            forecast_value += avg_drift
            self.forecasted_results.append(forecast_value)
            
        self.DriftDF = self.mergeDataFrame()
        
        if train_test == True:
            self.getEvaluation()
            self.getTrainForecastPlot()
        
        if save_model == True:
            self.saved_forecasts_.append(self.combined_df)
        
        
    def naiveSeasonalForecast(self,train_test = False, save_model = True, reset_ensemble_save_list = True):     
        
        self.model_name = 'NaiveSeasonal'
        if train_test == True:
            y = self.y_train 
        else:
            y = self.trimmed_data.Close
            
        self.y = y
        
        seasonality = sm.tsa.seasonal_decompose(y, period = 365).seasonal
        
        period = self.getPeriods()
        forecasted_results = np.repeat(y[-1], period)
        self.forecasted_results = (forecasted_results + seasonality[0:period])
        
        self.NaiveSeasonalDF = self.mergeDataFrame()
        
        if reset_ensemble_save_list == True:
            self.saved_forecasts_ = []
            
        if train_test == True:
            self.getEvaluation()
            self.getTrainForecastPlot()
        
        if save_model == True:
            self.saved_forecasts_.append(self.combined_df)
            
    def ensembleModelForecast(self):
        saved_dfs =pd.concat(self.saved_forecasts_)
        
        ensemble_forecast_value =round(saved_dfs[saved_dfs['Result']=='Forecasted'].pivot(columns = 'Model')['Close'].sum(axis = 1)/len(self.saved_forecasts_),2)
        
        df = pd.DataFrame(ensemble_forecast_value)
        df.columns = ['Close']
        df['Result'] = 'Forecasted'
        df['Model'] = 'Ensemble'
        df

        actuals_df =saved_dfs[saved_dfs['Result']=='Actual'].drop_duplicates()
        self.combined_df = pd.concat([df,actuals_df])

        
########### Train Forecast Methods #############3        
                   
    def ensembleModelTrainForecast(self):
        saved_dfs =pd.concat(self.saved_forecasts_)
        
        ensemble_forecast_value =round(saved_dfs[saved_dfs['Result']=='Forecasted'].pivot(columns = 'Model')['Close'].sum(axis = 1)/len(self.saved_forecasts_),2)
        
        df = pd.DataFrame(ensemble_forecast_value)
        df.columns = ['Close']
        df['Result'] = 'Forecasted'
        df['Model'] = 'Ensemble'
        df

        actuals_df =saved_dfs[saved_dfs['Result']=='Actual'].drop_duplicates()
        self.combined_df = pd.concat([df,actuals_df])

        self.getEvaluation()
        self.getTrainForecastPlot()

        
############ DataFrame merging #################################    
    def mergeDataFrame(self):
        y = self.y 
        df = self.trimmed_data
        df['Result'] = 'Actual'
        df['Model'] = 'Actual'
        
        forecasted_results = self.forecasted_results
        start_date = y.index[-1] + relativedelta(days=1)
        model_name = self.model_name
        
        forecast_index = pd.date_range(start = start_date, periods=len(forecasted_results), freq = 'B')
        forecast_df = pd.DataFrame(data={'Close':forecasted_results})
        forecast_df.index = forecast_index
        forecast_df['Result'] = 'Forecasted'
        forecast_df['Model'] = model_name
        combined_df =pd.concat([df,forecast_df], axis = 0)
        combined_df['Close'] = np.round(combined_df['Close'],2)
        
        self.combined_df = combined_df[['Close','Result','Model']]
        
        return self.combined_df
    
    
    
################### visualization ###############        
    def forecastPlot(self):
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
        )
    
    def getTrainForecastPlot(self):
        df=self.combined_df
        x=self.x
        trimmed_data = self.trimmed_data
        
        a_df = df[df['Model']=='Actual']
        m_df = df[df['Model']!='Actual']
        

        last_actual_dt=x[-1]

        plt.figure(figsize=(20,10))
        plt.plot(x, trimmed_data.Close)
        plt.plot(m_df[m_df.index <= last_actual_dt].Close)
        plt.legend(labels = df['Model'].unique())
        plt.show()
    

    ### TrainTestMethod #########
    
    def train_test_split(self, data = None, test_size = 0.20):
        try:
            trimmed_data = self.trimmed_data
        except:
            trimmed_data = data
            
        x = trimmed_data.index
        y = trimmed_data.Close
        
        test_set = int(len(y)*test_size)

        x_test = x[-test_set::]
        x_train = x[~x.isin(x_test)]

        y_test = y[-test_set::]
        y_train = y[~y.isin(y_test)]
        
        self.x_test = x_test
        self.y_test = y_test
        
        self.x_train = x_train
        self.y_train = y_train 
        
        return
    
        
    def getEvaluation(self):
        train_forecast_df = self.combined_df
        y_test = self.y_test
        
        resid =np.abs((train_forecast_df[train_forecast_df['Result']=='Forecasted']['Close'])-(y_test)).dropna()
        
        ## MAE
        MAE = np.sum(resid)/len(resid)
        print(f'MAE: {round(MAE,2)}')
        
        ## MAPE
        MAPE = np.sum((resid/y_test).dropna())/len(resid)
        print(f'MAPE: {round(MAPE*100,2)}%')
        

    
    def getPeriods(self):
        if self.y.index[-1] < self.trimmed_data.index[-1]:
            period = np.abs((self.y.index[-1] -self.trimmed_data.index[-1]).days)
        else:
            period = 365
        
        return period
    

        
        