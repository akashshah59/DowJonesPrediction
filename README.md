
# The Project aims to predict the change in  DIA stock values over time 


```python
import pandas as pd
import os
import seaborn as sns
```


```python
data = pd.read_csv('dow_jones_index.data')
data.dtypes
```




    quarter                                 int64
    stock                                  object
    date                                   object
    open                                   object
    high                                   object
    low                                    object
    close                                  object
    volume                                  int64
    percent_change_price                  float64
    percent_change_volume_over_last_wk    float64
    previous_weeks_volume                 float64
    next_weeks_open                        object
    next_weeks_close                       object
    percent_change_next_weeks_price       float64
    days_to_next_dividend                   int64
    percent_return_next_dividend          float64
    dtype: object




```python
data['quarter'].unique()
```




    array([1, 2])




```python
data.shape
#25 weeks per stock.
```




    (750, 16)



Data is spread over 2 quarters.


```python
# In[142]:
stocks = data['stock'].unique()
print("Different stock names : ",stocks)
print("Number of different stocks : ",len(stocks))
#30 Different types of stocks
```

    ('Different stock names : ', array(['AA', 'AXP', 'BA', 'BAC', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GE',
           'HD', 'HPQ', 'IBM', 'INTC', 'JNJ', 'JPM', 'KRFT', 'KO', 'MCD',
           'MMM', 'MRK', 'MSFT', 'PFE', 'PG', 'T', 'TRV', 'UTX', 'VZ', 'WMT',
           'XOM'], dtype=object))
    ('Number of different stocks : ', 30)


There are 30 stocks in the Dow Jones index


```python
import datetime as dt
```

Changing datatypes to suit our needs


```python
data['date'] = pd.to_datetime(data['date'])
data['stock'] = data['stock'].apply(lambda x: str(x))
data['open'] = data['open'].apply(lambda x:float(x.replace('$','')))
data['close'] = data['close'].apply(lambda x:float(x.replace('$','')))
data['high'] = data['high'].apply(lambda x:float(x.replace('$','')))
data['low'] = data['low'].apply(lambda x:float(x.replace('$','')))
data['next_weeks_open'] = data['next_weeks_open'].apply(lambda x:float(x.replace('$','')))
data['next_weeks_close'] = data['next_weeks_close'].apply(lambda x:float(x.replace('$','')))
```

Setting index to date type so as to visualize using **sns.tsplot**


```python
#dataplot = data.set_index('date',inplace = False)
%matplotlib inline
get_ipython().magic('matplotlib inline')
g = sns.FacetGrid(data,row = "stock" , col="quarter")
g.map(sns.tsplot,'percent_change_next_weeks_price')
g.add_legend()
```

    /Users/akash/anaconda2/lib/python2.7/site-packages/seaborn/timeseries.py:183: UserWarning: The tsplot function is deprecated and will be removed or replaced (in a substantially altered version) in a future release.
      warnings.warn(msg, UserWarning)





    <seaborn.axisgrid.FacetGrid at 0x10e04fad0>




![png](output_12_2.png)


The Face grid gives quarter wise trends of each stock.
Notice how every stock increases towards the end of quarter 2.
Let's check what the combined DIA index trend has to say.

## Fitting a Time series model and Forecasting index trends over time (Preliminary Analysis)


```python
pivoted = data.pivot(index= 'date',columns = 'stock')[['open','close']]
pivoted.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">open</th>
      <th>...</th>
      <th colspan="10" halign="left">close</th>
    </tr>
    <tr>
      <th>stock</th>
      <th>AA</th>
      <th>AXP</th>
      <th>BA</th>
      <th>BAC</th>
      <th>CAT</th>
      <th>CSCO</th>
      <th>CVX</th>
      <th>DD</th>
      <th>DIS</th>
      <th>GE</th>
      <th>...</th>
      <th>MRK</th>
      <th>MSFT</th>
      <th>PFE</th>
      <th>PG</th>
      <th>T</th>
      <th>TRV</th>
      <th>UTX</th>
      <th>VZ</th>
      <th>WMT</th>
      <th>XOM</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-07</th>
      <td>15.82</td>
      <td>43.30</td>
      <td>66.15</td>
      <td>13.85</td>
      <td>94.38</td>
      <td>20.45</td>
      <td>91.66</td>
      <td>50.05</td>
      <td>37.74</td>
      <td>18.49</td>
      <td>...</td>
      <td>37.35</td>
      <td>28.60</td>
      <td>18.34</td>
      <td>64.50</td>
      <td>28.85</td>
      <td>53.33</td>
      <td>79.08</td>
      <td>35.93</td>
      <td>54.08</td>
      <td>75.59</td>
    </tr>
    <tr>
      <th>2011-01-14</th>
      <td>16.71</td>
      <td>44.20</td>
      <td>69.42</td>
      <td>14.17</td>
      <td>93.21</td>
      <td>20.94</td>
      <td>90.95</td>
      <td>48.30</td>
      <td>39.01</td>
      <td>18.61</td>
      <td>...</td>
      <td>34.23</td>
      <td>28.30</td>
      <td>18.34</td>
      <td>65.53</td>
      <td>28.43</td>
      <td>54.63</td>
      <td>79.08</td>
      <td>35.46</td>
      <td>54.81</td>
      <td>77.84</td>
    </tr>
    <tr>
      <th>2011-01-21</th>
      <td>16.19</td>
      <td>46.03</td>
      <td>70.86</td>
      <td>15.08</td>
      <td>94.16</td>
      <td>21.22</td>
      <td>92.94</td>
      <td>49.53</td>
      <td>39.07</td>
      <td>18.98</td>
      <td>...</td>
      <td>33.90</td>
      <td>28.02</td>
      <td>18.36</td>
      <td>65.91</td>
      <td>28.33</td>
      <td>55.00</td>
      <td>80.20</td>
      <td>34.95</td>
      <td>55.73</td>
      <td>78.98</td>
    </tr>
    <tr>
      <th>2011-01-28</th>
      <td>15.87</td>
      <td>46.05</td>
      <td>71.52</td>
      <td>14.25</td>
      <td>92.71</td>
      <td>20.84</td>
      <td>93.89</td>
      <td>48.44</td>
      <td>39.64</td>
      <td>19.93</td>
      <td>...</td>
      <td>33.07</td>
      <td>27.75</td>
      <td>18.15</td>
      <td>64.20</td>
      <td>27.49</td>
      <td>55.81</td>
      <td>81.43</td>
      <td>35.63</td>
      <td>56.70</td>
      <td>78.99</td>
    </tr>
    <tr>
      <th>2011-02-04</th>
      <td>16.18</td>
      <td>44.13</td>
      <td>69.26</td>
      <td>13.71</td>
      <td>96.13</td>
      <td>20.93</td>
      <td>93.85</td>
      <td>50.15</td>
      <td>39.04</td>
      <td>20.13</td>
      <td>...</td>
      <td>32.89</td>
      <td>27.77</td>
      <td>19.30</td>
      <td>63.61</td>
      <td>27.97</td>
      <td>57.41</td>
      <td>82.52</td>
      <td>36.31</td>
      <td>56.03</td>
      <td>83.28</td>
    </tr>
    <tr>
      <th>2011-02-11</th>
      <td>17.33</td>
      <td>43.96</td>
      <td>71.43</td>
      <td>14.51</td>
      <td>99.62</td>
      <td>22.11</td>
      <td>97.28</td>
      <td>52.62</td>
      <td>40.80</td>
      <td>20.77</td>
      <td>...</td>
      <td>33.07</td>
      <td>27.25</td>
      <td>18.83</td>
      <td>64.73</td>
      <td>28.47</td>
      <td>58.99</td>
      <td>85.20</td>
      <td>36.39</td>
      <td>55.69</td>
      <td>82.82</td>
    </tr>
    <tr>
      <th>2011-02-18</th>
      <td>17.39</td>
      <td>46.42</td>
      <td>72.70</td>
      <td>14.77</td>
      <td>103.56</td>
      <td>18.84</td>
      <td>95.50</td>
      <td>54.44</td>
      <td>43.19</td>
      <td>21.51</td>
      <td>...</td>
      <td>32.85</td>
      <td>27.06</td>
      <td>19.19</td>
      <td>64.30</td>
      <td>28.57</td>
      <td>60.92</td>
      <td>85.01</td>
      <td>36.62</td>
      <td>55.38</td>
      <td>84.50</td>
    </tr>
    <tr>
      <th>2011-02-25</th>
      <td>16.98</td>
      <td>44.94</td>
      <td>72.35</td>
      <td>14.38</td>
      <td>104.86</td>
      <td>18.73</td>
      <td>99.23</td>
      <td>54.95</td>
      <td>42.83</td>
      <td>20.88</td>
      <td>...</td>
      <td>32.19</td>
      <td>26.55</td>
      <td>18.86</td>
      <td>62.84</td>
      <td>28.13</td>
      <td>59.60</td>
      <td>83.37</td>
      <td>35.97</td>
      <td>51.75</td>
      <td>85.34</td>
    </tr>
    <tr>
      <th>2011-03-04</th>
      <td>16.81</td>
      <td>43.73</td>
      <td>72.47</td>
      <td>14.27</td>
      <td>102.72</td>
      <td>18.62</td>
      <td>102.28</td>
      <td>54.22</td>
      <td>43.02</td>
      <td>20.95</td>
      <td>...</td>
      <td>33.06</td>
      <td>25.95</td>
      <td>19.66</td>
      <td>62.03</td>
      <td>27.92</td>
      <td>59.18</td>
      <td>82.86</td>
      <td>36.08</td>
      <td>52.07</td>
      <td>85.08</td>
    </tr>
    <tr>
      <th>2011-03-11</th>
      <td>16.58</td>
      <td>43.86</td>
      <td>71.60</td>
      <td>14.18</td>
      <td>103.42</td>
      <td>18.36</td>
      <td>104.12</td>
      <td>54.10</td>
      <td>43.53</td>
      <td>20.40</td>
      <td>...</td>
      <td>32.73</td>
      <td>25.68</td>
      <td>19.47</td>
      <td>61.49</td>
      <td>28.46</td>
      <td>58.88</td>
      <td>81.28</td>
      <td>35.85</td>
      <td>52.59</td>
      <td>82.12</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 60 columns</p>
</div>



### Pivoted dataframe makes it easy for us to calculate Dow Jones at opening and close


```python
pivoted['dow_jones_open'] = pivoted['open'].sum(axis = 1)
pivoted['dow_jones_close'] = pivoted['close'].sum(axis = 1)
```


```python
dow_jones = pivoted[['dow_jones_open','dow_jones_close']]
dow_jones.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>dow_jones_open</th>
      <th>dow_jones_close</th>
    </tr>
    <tr>
      <th>stock</th>
      <th></th>
      <th></th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-01-07</th>
      <td>1540.05</td>
      <td>1542.60</td>
    </tr>
    <tr>
      <th>2011-01-14</th>
      <td>1534.71</td>
      <td>1557.46</td>
    </tr>
    <tr>
      <th>2011-01-21</th>
      <td>1558.32</td>
      <td>1568.61</td>
    </tr>
    <tr>
      <th>2011-01-28</th>
      <td>1567.45</td>
      <td>1562.25</td>
    </tr>
    <tr>
      <th>2011-02-04</th>
      <td>1566.57</td>
      <td>1597.70</td>
    </tr>
  </tbody>
</table>
</div>




```python
import fbprophet,matplotlib
```


```python
dow_jones = dow_jones.reset_index()
dow_jones_open = dow_jones[['date','dow_jones_open']]
dow_jones_close = dow_jones[['date','dow_jones_close']]
dow_jones_open.rename(columns={'date': 'ds', 'dow_jones_open': 'y'}, inplace=True)
dow_jones_close.rename(columns={'date': 'ds', 'dow_jones_close': 'y'}, inplace=True)
```

    /Users/akash/anaconda2/lib/python2.7/site-packages/pandas/core/frame.py:4025: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    



```python
prophet = fbprophet.Prophet()
prophet.fit(dow_jones_close)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-27-60969e53ac76> in <module>()
    ----> 1 prophet = fbprophet.Prophet(weekly = False)
          2 prophet.fit(dow_jones_close)


    TypeError: __init__() got an unexpected keyword argument 'weekly'



```python
future = prophet.make_future_dataframe(periods=12,freq = 'W',include_history = True)
forecast = prophet.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>2011-08-14</td>
      <td>-329.840201</td>
      <td>-361.131709</td>
      <td>-300.482024</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2011-08-21</td>
      <td>-342.450380</td>
      <td>-377.288955</td>
      <td>-312.313497</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2011-08-28</td>
      <td>-355.060559</td>
      <td>-393.327032</td>
      <td>-319.291723</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2011-09-04</td>
      <td>-367.670738</td>
      <td>-408.731198</td>
      <td>-330.398371</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2011-09-11</td>
      <td>-380.280916</td>
      <td>-422.825414</td>
      <td>-337.204747</td>
    </tr>
  </tbody>
</table>
</div>



Plotting the forecasts with plotly


```python
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(prophet, forecast)
py.iplot(fig)
```


<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script><script type='text/javascript'>if(!window._Plotly){define('plotly', function(require, exports, module) {/**
* plotly.js v1.44.3
* Copyright 2012-2019, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/