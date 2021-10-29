import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report


n = 10

df = yf.download('AAPL', period = '1d', interval = '1m')
df = df.drop(df[df['Volume'] == 0].index)
df['RSI'] = ta.RSI(np.array(df['Close'].shift(1)),timeperiod= n)
df['SMA'] = df['Close'].shift(1).rolling(window=n).mean()
df['Corr'] = df['Close'].shift(1).rolling(window=n).corr(df['SMA'].shift(1))
df['SAR'] = ta.SAR(np.array(df['High'].shift(1)), np.array(df['Low'].shift(1)),
                   0.2, 0.2)
df['ADX'] = ta.ADX(np.array(df['High'].shift(1)), np.array(df['Low'].shift(1)),
                   np.array(df['Close'].shift(1)), timeperiod=n)
df['ATR'] = ta.ATR(np.array(df['High'].shift(1)), np.array(df['Low'].shift(1)),
                   np.array(df['Close'].shift(1)), timeperiod=n)
df['PH'] = df['High'].shift(1)
df['PL'] = df['Low'].shift(1)
df['PC'] = df['Close'].shift(1)
df['O-O']= df['Open'] - df['Open'].shift(1)
df['O-C']= df['Open'] - df['PC'].shift(1)
df['Ret'] = (df['Open'].shift(-1)-df['Open'])/df['Open']

for i in range(1, n):
    df['r%i' % i] = df['Ret'].shift(i)

df = df.dropna()


t = 0.8
split = int(t*len(df))

df['Signal'] = 0
df.loc[df['Ret'] > df['Ret'][:split].quantile(q=0.66), 'Signal'] = 1
df.loc[df['Ret'] < df['Ret'][:split].quantile(q=0.34), 'Signal'] = -1

X = df.drop(['Close','Signal','High','Low','Volume','Ret'], axis=1)
y = df['Signal']

c = [10,100,1000,10000]
g = [1e-2,1e-1,1e0]
p = {'svc__C': c, 'svc__gamma': g, 'svc__kernel': ['rbf']}
s = [('s', StandardScaler()), ('svc',SVC())]
pp = Pipeline(s)
rcv = RandomizedSearchCV(pp, p, cv = TimeSeriesSplit(n_splits=2))
rcv.fit(X.iloc[:split], y.iloc[:split])

c = rcv.best_params_['svc__C']
k = rcv.best_params_['svc__kernel']
g = rcv.best_params_['svc__gamma']

cls = SVC(C = c, kernel = k, gamma = g)
S = StandardScaler()
cls.fit(S.fit_transform(X.iloc[:split]), y.iloc[:split])

y_predict = cls.predict(S.transform(X.iloc[split:]))

df['Pred_Signal'] = 0

df.iloc[:split, df.columns.get_loc('Pred_Signal')] = pd.Series(
    cls.predict(S.transform(X.iloc[:split])).tolist())

df.iloc[split:, df.columns.get_loc('Pred_Signal')] = y_predict

df['Ret1'] = df['Ret']*df['Pred_Signal']

cr = classification_report(y[split:], y_predict)
cm = confusion_matrix(y[split:],y_predict)
print(cr)
print(cm)
