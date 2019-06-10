
# coding: utf-8

# In[1]:


# %load Dow+Jones+Prediction.py
import pandas as pd
import os
import seaborn as sns


# In[2]:


data = pd.read_csv('Downloads/dow_jones_index.data')
data.head(5)
data.dtypes


# In[3]:


#Data is distributed over 2 quarters
data['quarter'].unique()


# In[4]:


# In[142]:
stocks = data['stock'].unique()
print("Different stock names : ",stocks)
print("Number of different stocks : ",len(stocks))
#30 Different types of stocks


# In[5]:


import datetime as dt


# In[6]:


#Changing Data type of columns
data['date'] = pd.to_datetime(data['date'])
data['open'] = data['open'].apply(lambda x:float(x.replace('$','')))
data['close'] = data['close'].apply(lambda x:float(x.replace('$','')))
data['high'] = data['high'].apply(lambda x:float(x.replace('$','')))
data['low'] = data['low'].apply(lambda x:float(x.replace('$','')))
data['next_weeks_open'] = data['next_weeks_open'].apply(lambda x:float(x.replace('$','')))
data['next_weeks_close'] = data['next_weeks_close'].apply(lambda x:float(x.replace('$','')))


# In[7]:


data.set_index('date')
#inplace is False by default


# In[ ]:


get_ipython().magic(u'matplotlib inline')
get_ipython().magic('matplotlib inline')
g = sns.FacetGrid(data, row = "stock" , col="quarter")
g.map(sns.tsplot,'volume')
g.add_legend()


# In[8]:


pivoted = data.pivot(index= 'date',columns = 'stock')[['open','close']]


# In[9]:


#What does the Dow Jones look like now?
pivoted.head(2)


# In[10]:


pivoted['dow_jones_open'] = pivoted['open'].sum(axis = 1)
pivoted['dow_jones_close'] = pivoted['close'].sum(axis = 1)


# In[11]:


dow_jones = pivoted[['dow_jones_open','dow_jones_close']]


# In[12]:


import fbprophet,matplotlib


# In[13]:


dow_jones = dow_jones.reset_index()
dow_jones_open = dow_jones[['date','dow_jones_open']]
dow_jones_close = dow_jones[['date','dow_jones_close']]


# In[14]:


dow_jones_open.rename(columns={'date': 'ds', 'dow_jones_open': 'y'}, inplace=True)
dow_jones_open.rename(columns={'date': 'ds', 'dow_jones_close': 'y'}, inplace=True)


# In[ ]:


#Fitting model without Box Cox transformation
'''m = fbprophet.Prophet(daily_seasonality = False,yearly_seasonality = False,weekly_seasonality = False)
m.fit(dow_jones_open)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.plot(fig)'''


# In[15]:


m = fbprophet.Prophet(daily_seasonality = False,yearly_seasonality = False,weekly_seasonality = False)
m.fit(dow_jones_open)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast)


# In[16]:


#Too much variance in the results. Applying Box Cox transformations to get better results
from scipy.stats import boxcox

dow_jones_open_bc = dow_jones_open

dow_jones_open_bc['y'], lam = boxcox(dow_jones_open['y'])


# In[17]:


#What is our Lambda ?
lam


# In[18]:


new_m = fbprophet.Prophet(daily_seasonality = False,yearly_seasonality = False,weekly_seasonality = False)
new_m.fit(dow_jones_open_bc)


# In[19]:


future_bc = new_m.make_future_dataframe(periods=365)
forecast_bc = new_m.predict(future_bc)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

#These predictions would be wrong. Box Cox transformation must be applied.
#fig = plot_plotly(new_m, forecast)  # This returns a plotly Figure
#py.plot(fig)

new_m.plot(forecast_bc)


# In[20]:


forecast['yhat'].var()


# In[21]:


import plotly.graph_objs as go

fig = [go.Histogram(x = list(forecast['yhat']))]

py.plot(fig)


# In[24]:


from scipy.special import inv_boxcox


# In[25]:


inverted_forecast = forecast_bc[['yhat','yhat_upper','yhat_lower']].apply(lambda x: inv_boxcox(x, lam))
inverted_forecast['ds'] = forecast_bc['ds']


# In[26]:


#Checking Distribution of inverted_forecast 

fig = [go.Histogram(x = list(inverted_forecast['yhat']))]

py.plot(fig)


# In[ ]:


#fig = plot_plotly(new_m, inverted_forecast)  
#This returns a plotly Figure
#py.plot(fig)


# In[27]:


new_m.plot(inverted_forecast)
#Expected, can't use with new_m


# In[28]:


help(fbprophet.Prophet)


# In[ ]:


help(plot_plotly)


# In[ ]:


help(new_m)


# In[29]:


import plotly.graph_objs as go

data = [go.Scatter(x=inverted_forecast.ds, y=inverted_forecast['yhat'])]

py.plot(data, filename = 'time-series-simple')


# In[30]:


forecast.columns


# In[31]:


inverted_forecast.columns


# In[33]:


inverted_forecast['yhat'] - forecast['yhat']

data = [go.Scatter(x=inverted_forecast.ds, y=inverted_forecast['yhat'] - forecast['yhat'])]

py.plot(data, filename = 'Error comparisons.html')

'''We can see that as time progresses inverted_forecasts are a little less biased towards the left. 
Although stocks in both places show downwards trends. '''

