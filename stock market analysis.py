import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
