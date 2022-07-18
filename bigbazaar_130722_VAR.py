import pandas as pd
from pmdarima.arima import auto_arima
import numpy as np
from json import load
import pickle
import matplotlib.pyplot as plt


#-------------Reading the config file
with open("C:\\Users\\risha\\OneDrive\\Desktop\\DS\\Project - Innodatatics\\Client work\\config1.json", "r") as config_f:
    config = load(config_f)


df = config['bigbazar_data']
df=pd.read_csv(df)
datecolumn = config['datecolumn1']
pfilter = config['pfilter1']
freq = config['freq1']
dropcols = config['dropcols1']
mrp = config['mrp1']
disc = config['disc1']
price_gap = config['price_gap1']

val_counts = df['MC Description'].value_counts()
val_counts

df.info()

df[datecolumn] = df[datecolumn].apply(pd.to_datetime)

df1 = df.loc[df['MC Description'] == pfilter]
df1.reset_index(drop=True , inplace=True)

df1.plot(x="Netrate", y="Sum of Nsu", kind="line", figsize=(9, 8))
df1.plot(x="Netrate", y="Sum of Nsv", kind="line", figsize=(9, 8))

#df1 = df1[['Sum of Nsu','Netrate']]

df1 = df1.resample(freq, label='right', closed = 'right', on=datecolumn).sum().reset_index().sort_values(by=datecolumn)
#df1_log =df1_log[~df1_log.isin([np.nan, np.inf, -np.inf]).any(1)]

df1 = df1.sort_values(datecolumn)
df1.set_index(datecolumn, inplace=True)

df1.drop(dropcols,axis=1, inplace= True)

if freq == 'M':
    p = 12
if freq == 'Q':
    p = 4

flag=0
if ((df1.Netrate == 0).sum() != 0):
    df1['Sum of Nsu'] = df1['Sum of Nsu'] + 1
    df1['Netrate'] = df1['Netrate'] + 1
    flag+=1

df1_log = df1.copy()
df1_log = np.log10(df1_log)

from statsmodels.tsa.seasonal import seasonal_decompose
decompose_data = seasonal_decompose(df1_log.iloc[:,0],period=p, model="additive")
decompose_data.plot();
#AUTO ARIMA

exogdf = df1_log[['Netrate']]

arima_model = auto_arima(df1_log.iloc[:,0],X= exogdf,m=p,seasonal=True,start_p=0,start_q=0,max_order=4,test='adf',error_action='ignore',suppress_warnings=True,stepwise=True,trace=True)
arima_model.summary()


#For evaluation purpose
if freq == 'M':
    train = df1_log.iloc[:-5,:]
    test = df1_log.iloc[-5:,:]
    exogtrain = exogdf.iloc[:-5,:]
    exogtest = exogdf.iloc[-5:,:]
    
if freq == 'Q':
    train = df1_log.iloc[:-2,:]
    test = df1_log.iloc[-2:,:]
    exogtrain = exogdf.iloc[:-2,:]
    exogtest = exogdf.iloc[-2:,:]


arima_model.fit(train['Sum of Nsu'],X = exogtrain)

#arima_model.plot_diagnostics(figsize=(18, 8))


#exogtest.Netrate = np.log10(49)
forecast = arima_model.predict(n_periods=len(test),X = exogtest,return_conf_int=True)
pred_df = pd.DataFrame(forecast[0],index = test.index,columns=['Prediction'])


pd.concat([df1_log['Sum of Nsu'],pred_df],axis=1).plot(figsize=(16,9))
pd.concat([10 ** df1_log['Sum of Nsu'], 10 ** pred_df],axis=1).plot(figsize=(16,9))

testr = (10 ** test['Sum of Nsu'])
predr = (10 ** pred_df['Prediction'])
if (flag==1):
    testr = testr - 1
    predr = predr -1
error = (testr - predr)
errorp = (error/ testr) *100
MAE = np.abs(error).mean()
MAPE = np.abs(errorp).mean()
print(MAE)
print(MAPE)



#Saving model
filename = 'finalized_model.sav'
pickle.dump(arima_model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))



# INPUT MRP FROM JSON
# INPUT DISCOUNT FROM JSON
# INPUT PRICE GAP FROM JSON
val = mrp * disc
price = np.arange(mrp-val,mrp+val+1, price_gap)
price

testprice = pd.DataFrame(np.zeros(len(test)),columns=['Netrate'])
predprice = pd.DataFrame()
for i in price:
    testprice['Netrate'] = np.log10(i)
    p = str(i)
    forecast = arima_model.predict(n_periods=len(test),X = testprice,return_conf_int=True)
    predprice[p] = forecast[0]
predprice = 10 ** (predprice)

# Best Price 
bestprice = float(predprice.idxmax(axis=1).mode())


revenue = pd.DataFrame()
for i in predprice.columns:
    p = float(i)
    revenue[i] = p * predprice[i]

# Best Revenue
bestrev_price = revenue.idxmax(axis=1).mode()[0]
result_revenue = revenue[bestrev_price]



pred_nsu = predprice.loc[0]

plt.plot(price, pred_nsu)
plt.xlabel('Price')
plt.ylabel('Predicted NSU')

plt.plot(test.index, result_revenue)
plt.xlabel('Timestamp')
plt.ylabel('Revenue for each step')

#plt.plot(price, (0.3*result_revenue))
#plt.xlabel('Price')
#plt.ylabel('Margin')

# importing mplot3d toolkits, numpy and matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
 
fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
 
# defining all 3 axes
z = result_revenue
x = price
y = pred_nsu
 
# plotting
ax.plot3D(x, y, z, 'green')
ax.set_title('3D plot')
plt.show()



'''
################### VAR ########################




from statsmodels.tsa.stattools import grangercausalitytests

#Performing test on for realgdp and realcons.
gc_res = grangercausalitytests(df1_log, 10)


from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(data, alpha=0.05): 

   """Perform Johanson's Cointegration Test and Report Summary"""
   out = coint_johansen(data,-1,5)
   d = {'0.90':0, '0.95':1, '0.99':2}
   traces = out.lr1
   cvts = out.cvt[:, d[str(1-alpha)]]
   def adjust(val, length= 6): return str(val).ljust(length)

   # Summary
   print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
   for col, trace, cvt in zip(data.columns, traces, cvts):
       print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
cointegration_test(df1_log)
# Cointegration fail as not greater than 95% 

from statsmodels.tsa.api import VAR

var = VAR(train)
x = var.select_order(maxlags=8)
y= x.summary()
print(y.as_text())

for i in [1,2,3,4,5,6,7,8,9]:
    result = var.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

model_fitted = var.fit(9)
model_fitted.summary()

from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

def adjust(val, length= 6): return str(val).ljust(length)

for col, val in zip(df1_log.columns, out):
    print(adjust(col), ':', round(val, 2))


# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df1_log.values[-lag_order:]
forecast_input

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=5)
df_forecast = pd.DataFrame(fc, index=test.index, columns=test.columns)
df_forecast


'''