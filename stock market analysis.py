# %% Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlibhelper as mh
import matplotlib.pyplot as plt
import scipy.stats as stats

from pandas import Series, DataFrame
from pandas_datareader import DataReader
from datetime import datetime

sns.set_style('whitegrid')

# %% Collect and describe data
"""
First we collect the financial data from Apple (AAPL), Google (GOOG), Microsoft (MSFT) and Amazon (AMZN).
Below is an example from AAPL of how the data is constructed.
"""

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()

start = datetime(end.year-1, end.month, end.day)

for stock in tech_list:
    globals()[stock]= DataReader(stock, 'yahoo', start, end)

AAPL.describe()

AAPL.info()

# %% Show Adj Close for AAPL of the past year

AAPL['Adj Close'].plot(legend = True, figsize= (20,4))

# %% Show Volume for AAPL of the past year

AAPL['Volume'].plot(legend=True, figsize=(20,4))

# %% Generate a moving average for Adj Close of Apple

ma_day = [10, 20, 50]

for ma in ma_day:
    column_name = "Moving Average for %s days" %(str(ma))
    AAPL[column_name] = pd.Series(AAPL["Adj Close"]).rolling(ma).mean()

AAPL[["Adj Close", "Moving Average for 10 days", "Moving Average for 20 days", "Moving Average for 50 days" ]].plot(legend = True, figsize = (20,4))

# %% Calculate the percentage change per day

AAPL["Daily Return"] = AAPL["Adj Close"].pct_change()

AAPL["Daily Return"].plot(figsize = (20,4), legend = True, linestyle = "--", marker = "o")

# %% Give a distribution plot of the Daily Return

fig = plt.figure(figsize = (20,4))
sns.distplot(AAPL["Daily Return"].dropna(), bins = 100, color = 'blue')

# %% Alternative way for making a distribution plot

AAPL["Daily Return"].hist(bins = 100, figsize = (20,4))

# %% New datasets of Adj Close
closing_df = DataReader(tech_list, 'yahoo', start, end)['Adj Close']
closing_df.head()

# %% New dataset of percent change
tech_rets = closing_df.pct_change()
tech_rets.head()

# %% Make a jointplot of GOOG vs GOOG

sns.jointplot('GOOG', 'GOOG', tech_rets, kind = 'scatter', color = 'seagreen')

# %% Make a jointplot of GOOG VS AMZN
fig = sns.jointplot('GOOG', 'AMZN', tech_rets, kind = 'scatter', color = 'blue')
fig.annotate(stats.pearsonr)

# %% Display tech_rets

tech_rets.head()

# %% show pairplots

sns.pairplot(tech_rets.dropna())

# %% alternative way of making pairplots

returns_fig = sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter, color = 'green')
returns_fig.map_lower(sns.kdeplot, cmap = 'cool_d')
returns_fig.map_diag(plt.hist, bins = 30)
