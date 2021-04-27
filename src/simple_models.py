import os
import itertools
import pandas as pd
import numpy as np
import datetime
import scipy.stats as stats
from scipy import signal
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm 
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from cleaning import merge_dfs

'''
linear regression
* simple
* regularized
* predictive
* multivariate
ridge
lasso
elastic net

time series
'''

def load_data():
    df = pd.read_csv('../data/processed/sales.csv',parse_dates=['Date'])
    df = df[['Customer ID', 'Cohort Yr','purchases', 'State','Document Number', 
            'Date', 'Year', 'Month', 'Sales Total', 'log_sales']]
    rfm = pd.read_csv('../data/processed/rfm.csv')
    rfm = rfm[['Customer ID','frequency','recency','T', 'monetary_value','prob_alive','pred_purchases']]
    df = merge_dfs(df, [rfm], key='Customer ID')
    df = df[['Customer ID', 'Cohort Yr','purchases', 'State',
            'frequency','recency','T', 'monetary_value','prob_alive','pred_purchases',
            'Document Number','Date', 'Year', 'Month','log_sales', 'Sales Total' ]]
    return df

def process_data(df):
    df['Document Number'] = df['Document Number'].map(lambda x: x.lstrip('INV'))
    df = df.join(pd.get_dummies(df['State']))
    df['Date'] = pd.to_numeric(pd.to_datetime(df['Date']))
    df.drop('State', axis=1, inplace=True)
    print(df.head(),'\n',df.shape,'\n')
    return df

def show_scatter(df):
    pd.plotting.scatter_matrix(df, figsize=(15,10))
    plt.show()
    return df

def LinReg_model(X_train, X_test, y_train, y_test):
    model =LinearRegression()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    print(yhat,'\n\n')
    mse = mean_squared_error(y_test, yhat)
    print('mse = ',mse)
    rmse = np.sqrt(mse)
    print('rmse = ', rmse, 'dollars')
    model=sm.OLS(y, X.astype(float))
    results = model.fit()
    print(results.summary())
    ypred = results.predict()
    print('Linear Regression prediction using OLS = ',ypred.sum())
    return results



def ARIMA_pre_processing():
    df = pd.read_csv('../data/processed/sales.csv',parse_dates=['Date'])
    series=df[['Date','Sales Total']]
    series.columns = ['Date','Sales']
    series['year_month']=pd.to_datetime(series['Date']).dt.to_period('M')
    series = pd.DataFrame(series.groupby('year_month')['Sales'].sum())
    # series.columns = ['Sales']
    # series = series[['year_month', 'Sales']].reset_index(drop=True).set_index('year_month')
    # series['Sales'].astype(int)
    trend = series['Sales']
    return [series, trend]

def plot_trend_data(ax,name, series):
    ax.plot(series.index, series)
    ax.set_title('Sales')
    ax.show()
    return ax

def plot_seasonal_decomposition(axs, series, sd):
    python_decomposition = sm.tsa.seasonal_decompose('Sales')
    axs[0].plot(series.index, series)
    axs[0].set_title("Raw Series")
    axs[1].plot(series.index, sd.trend)
    axs[1].set_title("Trend Component $T_t$")
    axs[2].plot(series.index, sd.seasonal)
    axs[2].set_title("Seasonal Component $S_t$")
    axs[3].plot(series.index, sd.resid)
    axs[3].set_title("Residual Component $R_t$")  
    fig, axs = plt.subplots(4, figsize=(14, 8))
    plot_seasonal_decomposition(axs, trend, python_decomposition)
    plt.tight_layout()
    return fig 

def plot_series_and_difference(axs, series, title='Sales'):
    diff = series.diff()
    axs[0].plot(series.index, series)
    axs[0].set_title("Raw Series: {}".format(title))
    axs[1].plot(series.index, diff)
    axs[1].set_title("Series of First Differences: {}".format(title)) 
    fig, axs = plt.subplots(2, figsize=(14, 4))
    # plot_series_and_difference(axs, trend, trend_of_interest)
    fig.tight_layout()
    return fig

def plot_autocorrelation():
    fig, ax - plt.subplots(1, figsize=(16,4))
    auto_plot = sm.graphics.tsa.plot_acf(yt, lags=2*12, ax=ax)  #yt=trend.diff()[1:]
    return auto_plot

# def ARIMA_pre_plotting():
#     # call plot functions above


def ARIMA_model(trend,p, d, q):
    model = ARIMA(trend, order=(p,d,q)).fit()
    print(model.summary())
    model.params[1:]
    print('\n ARIMA params \n',model.params[1:])
    # group sales data by month
    trend2 = trend.copy()
    trend2.index = pd.date_range(start='2015-01-01', end='2020-12-31', freq='M')

    fig, ax = plt.subplots(figsize=(12,6))
    ax = trend2.loc['2015-01-01':].plot(ax=ax)
    fig = model.plot_predict('2021-01-01', '2023', dynamic=True, ax=ax, plot_insample=False)
    plt.show()
    return model   


def plot_features(X, target):
    X_cols = X.columns 
    for col in X_cols:
        df.plot(kind='scatter', y=target, x=col, edgecolor='none', figsize=(12,5))
        plt.xlabel(col)
        plt.ylabel(target)
        plt.show()
    return plt

if __name__ == "__main__":
    df = load_data()
    df = process_data(df)
    #scatter_plots = show_scatter(df)
    
    # Set feature and target matrix
    X = df[['Customer ID', 'Cohort Yr', 'purchases', 'frequency', 'recency', 'T',
       'monetary_value', 'prob_alive', 'pred_purchases', 'Document Number',
       'Date', 'Year', 'Month', 'AK', 'CA', 'FL',
       'GA', 'ID', 'NH', 'NV', 'OK', 'OR', 'WA', 'WACA']]
    y = df['Sales Total']
    ## not used yet ['log_sales']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    # fit Linear Regression
    linreg = LinReg_model(X_train, X_test, y_train, y_test)
    # plots = plot_features(X, 'Sales Total')

    
    # Plot
    # df.plot(kind='scatter', y='Sales Total', x='monetary_value', figsize=(12,5))
    # plt.show()

    series, trend = ARIMA_pre_processing()
    print('trend',trend)


    arima = ARIMA_model(trend, p=2, d=1, q=0)


    annuals = df.groupby('Year')['Sales Total'].sum()

    annuals.index = pd.date_range(start='2015', end='2021', freq='Y')
    fig, ax = plt.subplots(figsize=(12,6))
    ax = annuals.loc['2015':].plot(ax=ax)
    fig = arima.plot_predict('2021', '2023', dynamic=True, ax=ax, plot_insample=False)
    plt.show()

    # arima.plot_predict(dynamic=True)
    # plt.show()



    



# remove QTY, Unit Price, Revenue
# add Age, Recency, Freq, Prob_Alive, Pred_freq
# sales = sales.groupby('Date')['Total Sales'].reset_index()  # or [year, month]
# sales = sales.set_index('Order Date')

# Scatter Sales and other data
# correlation of variables using heatmap in sns


