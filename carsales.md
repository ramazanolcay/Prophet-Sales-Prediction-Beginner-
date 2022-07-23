```python
import pandas as pd
import numpy as np
```


```python
data = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv")

import matplotlib.pyplot as plt

data.plot()
plt.show()
```


    
![png](output_1_0.png)
    



```python
data.head()
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
      <th>Month</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1960-01</td>
      <td>6550</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1960-02</td>
      <td>8728</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1960-03</td>
      <td>12026</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1960-04</td>
      <td>14395</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1960-05</td>
      <td>14587</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.dtypes
```




    Month    object
    Sales     int64
    dtype: object




```python
#Converting obhect months to datetime
data.columns = ["ds","y"]
data["ds"] = pd.to_datetime(data["ds"])
data.dtypes
```




    ds    datetime64[ns]
    y              int64
    dtype: object




```python
train = data.drop(data.index[-12:])
train.tail()
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
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>91</th>
      <td>1967-08-01</td>
      <td>13434</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1967-09-01</td>
      <td>13598</td>
    </tr>
    <tr>
      <th>93</th>
      <td>1967-10-01</td>
      <td>17187</td>
    </tr>
    <tr>
      <th>94</th>
      <td>1967-11-01</td>
      <td>16119</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1967-12-01</td>
      <td>13713</td>
    </tr>
  </tbody>
</table>
</div>




```python
#test values have been droped
from fbprophet import Prophet

mymodel = Prophet()
mymodel.fit(train)
```

    INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    C:\ProgramData\Anaconda3\lib\site-packages\fbprophet\forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(new_comp)
    




    <fbprophet.forecaster.Prophet at 0x1a94ceb5640>




```python
#creating future dates (1968)
future = list()

for i in range(1,13):
    date = "1968-%02d" % i
    future.append([date])
    
future = pd.DataFrame(future)
future.columns = ["ds"]
future["ds"] = pd.to_datetime(future["ds"])
future["ds"]
```




    0    1968-01-01
    1    1968-02-01
    2    1968-03-01
    3    1968-04-01
    4    1968-05-01
    5    1968-06-01
    6    1968-07-01
    7    1968-08-01
    8    1968-09-01
    9    1968-10-01
    10   1968-11-01
    11   1968-12-01
    Name: ds, dtype: datetime64[ns]




```python
#predicting
forecast = mymodel.predict(future)

y_test = data["y"][-12:].values
y_pred = forecast["yhat"].values
```

    C:\ProgramData\Anaconda3\lib\site-packages\fbprophet\forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(new_comp)
    C:\ProgramData\Anaconda3\lib\site-packages\fbprophet\forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(new_comp)
    


```python
#metrics R_2 and MSE
from sklearn.metrics import mean_absolute_error
mse = mean_absolute_error(y_test,y_pred)
print("mse: ",mse)

from sklearn.metrics import r2_score
my_score = r2_score(y_test,y_pred)
print("r2 score: ", my_score)
```

    mse:  1336.8137661823855
    r2 score:  0.7816998351995643
    


```python
plt.plot(y_test, label="Actuals")
plt.plot(y_pred, label="Predictions")
plt.legend()
plt.show()
```


    
![png](output_10_0.png)
    



```python

```
