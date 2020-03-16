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

AAPL.head()

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

plt.figure(figsize = (20,4))
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

# %% pairplots of closing prices

returns_fig = sns.PairGrid(closing_df.dropna())
returns_fig.map_upper(plt.scatter, color = 'green')
returns_fig.map_lower(sns.kdeplot, cmap = 'cool_d')
returns_fig.map_diag(plt.hist, bins = 30)

# %% correlation plot for daily returns

corr = tech_rets.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot = True, mask = mask)

# %% correlation plot for closing prices

corr = closing_df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, annot = True, mask = mask)

# %% scatterplot risk vs expected return

rets = tech_rets.dropna()
area = np.pi*10

plt.figure(figsize = (20,4))
plt.scatter(rets.mean(), rets.std(), s = 20)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(connectionstyle = 'arc3, rad = -0.3', arrowstyle = 'simple')
    )

# %% histogram of AAPL Daily Return

plt.figure(figsize = (20,4))
sns.distplot(AAPL['Daily Return'].dropna(), bins = 100, color = 'red')

# %% return the head of rets

rets.head()

# %% return the 0.05 quantile

print(rets['AAPL'].quantile(0.05))

# %% create monte carlo function

days = 365
dt = 1/days
mu = rets.mean()['AAPL']
sigma = rets.std()['AAPL']

def stock_monte_carlo(start_price, days, mu, sigma):
    price = np.zeros(days)
    price[0] = start_price

    shock = np.zeros(days)
    drift = np.zeros(days)

    for x in range(1, days):
        shock[x] = np.random.normal(loc = mu * dt, scale = sigma * np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))

    return price

# %% test simulation

AAPL.head()

start_price = 320

plt.figure(figsize = (20,8))

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Apple')

# %%

runs = 10000
simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]

# %%
q = np.percentile(simulations, 1)

plt.figure(figsize = (20,4))
plt.hist(simulations, bins = 200)
plt.figtext(0.6, 0.8, s = "Start price: $%.2f" %start_price)
plt.figtext(0.6, 0.7, s = "Mean final price: $%.2f" %simulations.mean())
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" %(start_price-q,))
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)
plt.axvline(x = q, linewidth = 4, color = 'r')
plt.title("Final price distribution for Apple Stock after %s days" %days, weight = 'bold')
