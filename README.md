              precision    recall  f1-score   support

          -1       0.89      0.86      0.87        28
           0       0.72      0.59      0.65        22
           1       0.76      0.92      0.83        24

    accuracy                           0.80        74
    macro avg      0.79      0.79      0.78        74
    weighted avg   0.80      0.80      0.79        74

[[24  3  1]<br>
 [ 3 13  6]<br>
 [ 0  2 22]]
---
# To install yfinance & talib on colab :<br>

```python
!pip install yfinance
```

```python
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
```

```python
!tar xvzf ta-lib-0.4.0-src.tar.gz
```

```python
import os
```

```python
os.chdir('ta-lib')
```

```python
!./configure --prefix=/usr
```

```python
!make
```

```python
!make install
```

```python
os.chdir('../')
```

```python
!pip install TA-Lib
```
