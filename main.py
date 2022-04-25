import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

from pmdarima.arima import auto_arima

from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
stock_data = pd.read_csv('acgl.txt',sep=',', index_col='Date', parse_dates=['Date'], date_parser=dateparse).fillna(0)

ptint(stock_data)

df_close = stock_data['Close']
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Дата')
plt.ylabel('Ціна закриття')
plt.plot(stock_data['Close'])
plt.title('ARCH CAPITAL GROUP ціна закриття')
plt.show()


def test_stationarity(timeseries):

    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    
    plt.plot(timeseries, color='blue',label='Оригінал')
    plt.plot(rolmean, color='red', label='Ковзне середнє значення')
    plt.plot(rolstd, color='black', label = 'Стандартне відхилення')
    plt.legend(loc='best')
    plt.title('Ковзне середнє значення та стандартне відхилення')
    plt.show(block=False)
    
    print("Результати тесту Дікі Фуллера")
    adft = adfuller(timeseries,autolag='AIC')
    
    output = pd.Series(adft[0:4],index=['Статистика тестів','p-значення','К-ть використаних lags','К-ть використаних спостережень'])
    for key,values in adft[4].items():
        output['критичне значення (%s)'%key] =  values
    print(output)
    
print(test_stationarity(df_close))


result = seasonal_decompose(df_close, model='multiplicative', freq = 30)
fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(16, 9)


from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
df_log = np.log(df_close)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc='best')
plt.title('Ковзне середнє')
plt.plot(std_dev, color ="black", label = "Стандартне відхилення")
plt.plot(moving_avg, color="red", label = "Середнє")
plt.legend()
plt.show()



train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Дата')
plt.ylabel('Ціна закриття')
plt.plot(df_log, 'green', label='Тренувальні дані')
plt.plot(test_data, 'blue', label='Тестувальні дані')
plt.legend()



model = ARIMA(train_data, order=(3, 1, 2))  
fitted = model.fit(disp=-1)  

fc, se, conf = fitted.forecast(321, alpha=0.05)
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='Тренувальні дані')
plt.plot(test_data, color = 'blue', label='Актуальна ціна акції')
plt.plot(fc_series, color = 'orange',label='Прогнрозована ціна акції')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.10)
plt.title('Прогнозування ціни акції ARCH CAPITAL GROUP')
plt.xlabel('Дата')
plt.ylabel('Ціна акції')
plt.legend(loc='upper left', fontsize=8)
plt.show()


mse = mean_squared_error(test_data, fc)
print('MSE: '+str(mse))
mae = mean_absolute_error(test_data, fc)
print('MAE: '+str(mae))
rmse = math.sqrt(mean_squared_error(test_data, fc))
print('RMSE: '+str(rmse))
mape = np.mean(np.abs(fc - test_data)/np.abs(test_data))
print('MAPE: '+str(mape))
