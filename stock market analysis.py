import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

from pandas import Series, DataFrame
from pandas_datareader import DataReader
from datetime import datetime

sns.set_style('whitegrid')

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

end = datetime.now()

start = datetime(end.year-1, end.month, end.day)

for stock in tech_list:
    globals()[stock]= DataReader(stock, 'yahoo', start, end)

AAPL.describe()

AAPL.info()

AAPL['Adj Close'].plot(legend = True, figsize= (20,4))

AAPL['Volume'].plot(legend=True, figsize=(20,4))

ma_day = [10, 20, 50]

for ma in ma_day:
    column_name = "Moving Average for %s days" %(str(ma))
    AAPL[column_name] = pd.Series(AAPL["Adj Close"]).rolling(ma).mean()

AAPL[["Adj Close", "Moving Average for 10 days", "Moving Average for 20 days", "Moving Average for 50 days" ]].plot(legend = True, figsize = (20,4))

AAPL["Daily Return"] = AAPL["Adj Close"].pct_change()

AAPL["Daily Return"].plot(figsize = (20,4), legend = True, linestyle = "--", marker = "o")

fig = plt.figure(figsize = (20,4))
sns.distplot(AAPL["Daily Return"].dropna(), bins = 100, color = 'blue')
AAPL["Daily Return"].hist(bins = 100, figsize = (20,4))

closing_df = DataReader(tech_list, 'yahoo', start, end)['Adj Close']
closing_df.head()

tech_rets = closing_df.pct_change()
tech_rets.head()

sns.jointplot('GOOG', 'GOOG', tech_rets, kind = 'scatter', color = 'seagreen')

fig = sns.jointplot('GOOG', 'AMZN', tech_rets, kind = 'scatter', color = 'blue')
fig.annotate(stats.pearsonr)
