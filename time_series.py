# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 20:12:03 2019

@author: USER
"""
#%% Import time series in python?
import pandas as pd

# Import as Dataframe
df = pd.read_csv('./data/a10.csv', parse_dates=['date'], index_col='date')
#df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', 
#                 parse_dates=['date'], index_col='date')
df.head()

#%% Visualizing a time series
import matplotlib.pyplot as plt

# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(8,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.index, y=df.value, 
        title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.')
#%% How to decompose a time series into its components?
from statsmodels.tsa.seasonal import seasonal_decompose

# Multiplicative Decomposition 
result_mul = seasonal_decompose(df['value'], 
                                model='multiplicative', 
                                extrapolate_trend='freq')
result_mul.seasonal.head(24)
#%% Plot
plt.rcParams.update({'figure.figsize': (6, 4), 'figure.dpi': 120})
result_mul.plot()
plt.show()

#%% Moving Averages in pandas
import pandas as pd

product = {'month' : [1,2,3,4,5,6,7,8,9,10,
                      11,12, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24],
           'demand':[143,152,161,139,137,174,142,141,162,180,
                     164,171,206,193,207,218,229,225,204,227,
                     223,242,239,266]}
df = pd.DataFrame(product)
df.head()
#%% Using the pandas in-built rolling function with Window size = 5 
df['SMA_5'] = df.iloc[:,1].rolling(window=5).mean()

# Using the pandas in-built rolling function with Window size = 10
df['SMA_10'] = df.iloc[:,1].rolling(window=10).mean()

df.head(15)
#%% Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=[8,5])
plt.grid(True)
plt.plot(df['demand'],label='data')
plt.plot(df['SMA_5'],label='SMA 5 Months', marker='o')
plt.plot(df['SMA_10'],label='SMA 10 Months', marker='x')
plt.legend(loc=2)
#%% Single Exponential Smoothing (SES)
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# Single Exponential Smoothing
fit1 = SimpleExpSmoothing(df['demand']).fit(smoothing_level=0.2, optimized=False)
fcast1 = fit1.forecast(4).rename(r'$\alpha=0.2$')

# plot
import matplotlib.pyplot as plt
plt.figure(figsize=[8,5])
plt.grid(True)
plt.plot(df['demand'],label='data')
plt.plot(fit1.fittedvalues,label=r'$\alpha=0.2$', marker='x')
plt.plot(fcast1,label='Forecast', marker='o')
plt.legend(loc=2)


#%% Double Expontial Smoothing (DES)
fit2 = Holt(df['demand']).fit(smoothing_level=0.2, 
           smoothing_slope=0.3, optimized=False)
fcast2 = fit2.forecast(4).rename(r'$\alpha=0.2$, $\beta=0.3$')

# plot
plt.figure(figsize=[8,5])
plt.grid(True)
plt.plot(df['demand'],label='data')
plt.plot(fit2.fittedvalues,label=r'$\alpha=0.2$, $\beta=0.3$', marker='x')
plt.plot(fcast2,label='Forecast', marker='o')
plt.legend(loc=2)

#%% Holt-Winters’ Method
# data with trend and seasonality
import numpy as np
sales = {'month' : [1,2,3,4,5,6,7,8,9,10,
                    11,12, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24],
         'demand':[362,385,432,341,382,409,498,387,473,513,
                   582,474,544,582,681,557,628,707,773,592,
                   627,725,854,661]}
df = pd.DataFrame(sales)

plt.figure(figsize=[8,5])
plt.grid(True)
plt.plot(df['demand'],label='data')
plt.legend(loc=2)
#%% Triple Exponential Smoothing(Holt-Winters’ Method)
fit3 = ExponentialSmoothing(df['demand'], seasonal_periods=4, 
                            trend='add', seasonal='add').fit(use_boxcox=True)
fcast3 = fit3.forecast(4).rename('seasonal_periods=4')

# plot
plt.figure(figsize=[8,5])
plt.grid(True)
plt.plot(df['demand'],label='data')
plt.plot(fit3.fittedvalues,label='seasonal_periods=4', marker='x')
plt.plot(fcast3,label='Forecast', marker='o')
plt.legend(loc=2)
#%% ########################################################
############################################################
####################  Excercise HW method ##################
############################################################
############################################################
#%% (Monthly anti-diabetic drug sales in Australia from 1992 to 2008)
import pandas as pd

# Import as Dataframe
df = pd.read_csv('./data/a10.csv', 
                 parse_dates=['date'], index_col='date')
df.head()
#%% Visualizing a time series
import matplotlib.pyplot as plt

# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(8,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(df, x=df.index, y=df.value, 
        title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.')

#%% Decompose a time series into its components
from statsmodels.tsa.seasonal import seasonal_decompose

# Multiplicative Decomposition 
result_mul = seasonal_decompose(df['value'], model='multiplicative', 
                                extrapolate_trend='freq')
result_mul.seasonal.head(24)
#%% Decomposition Plot
plt.rcParams.update({'figure.figsize': (6,4)})
result_mul.plot()
plt.show()

#%% Triple Exponential Smoothing(Holt-Winters’ Method)
fit3 = ExponentialSmoothing(df['value'], seasonal_periods=12, 
                            trend='add', seasonal='mul').fit(use_boxcox=True)
fcast3 = fit3.forecast(24).rename('seasonal_periods=12')

# plot
plt.figure(figsize=[8,5])
plt.grid(True)
plt.plot(df['value'],label='data')
plt.plot(fit3.fittedvalues,label='seasonal_periods=12', marker='x')
plt.plot(fcast3,label='Forecast', marker='o')
plt.legend(loc=2)
#%%#########################################################
################### End of Time Series   ###################
############################################################

