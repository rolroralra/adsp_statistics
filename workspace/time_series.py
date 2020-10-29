#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:50:27 2020

@author: shinyoungkim
"""
#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.getcwd();
os.chdir("./data")
os.getcwd();
#%%
###########################
# Visualizing Time Series
###########################

data = pd.read_csv('a10.csv', parse_dates=['date'], index_col='date')

def plot_df(df, x, y, title="Time Series", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(8,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.grid()
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
    
plot_df(data, x=data.index, y=data[data.columns[0]])
#%%
###############################
# Decomposition of Time Series
###############################
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_result = seasonal_decompose(data[data.columns[0]], model='multiplicative', extrapolate_trend='freq')
type(decompose_result)
dir(decompose_result)
print(decompose_result.seasonal.head())

plt.figure(figsize=(6,4), dpi=120)
decompose_result.plot()
plt.show()
#%%
###############################
# Moving Average
###############################

product = {'month': np.arange(1,25), 'demand': [143,152,161,139,137,174,142,141,162,180,
                                                164,171,206,193,207,218,229,225,204,227,
                                                223,242,239,266]}
data = pd.DataFrame(product)

data['MA_5'] = data.iloc[:,1].rolling(window=5).mean()
data['MA_10'] = data.iloc[:,1].rolling(window=10).mean()

print(data.head(15))
#%%%
