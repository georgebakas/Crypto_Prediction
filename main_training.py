import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime as datetime
import plotly.express as px

import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from sklearn import preprocessing
from joblib import dump, load

# Data imported from: http://www.cryptodatadownload.com/data/binance/
df = pd.read_csv('data/Binance_BTCUSDT_d.csv', skiprows=[0])
print(df.dtypes)

print("\n*** DF ***")
df_copy = df.copy()

def change_timestamp (ts):
    digit_count = len(str(ts))
    if digit_count == 12:
        return (datetime.datetime.utcfromtimestamp(ts)).strftime('%Y-%m-%d %H:%M:%S')
    else:
        return (datetime.datetime.utcfromtimestamp(ts/1000)).strftime('%Y-%m-%d %H:%M:%S')

df_copy['unix_count'] = df.unix.apply(lambda x: len(str(x)))        
df_copy['dt_correct'] = df.unix.apply(lambda x: change_timestamp(x))
df_copy['dt'] = pd.to_datetime(df_copy.dt_correct.values)
df_copy['hour'] = df_copy.dt.apply(lambda x: x.hour)
df_copy['week_day'] = df_copy.dt.apply(lambda x: x.weekday())
df_copy.sort_values(by=['unix'],ascending=[True],inplace=True)

df_work = df_copy[['dt','hour','week_day','close','Volume BTC']]
plt.figure(figsize=(15,7))
sns.set()
sns.lineplot(x = df_work.dt, y = 'close',data=df_work).set_title('Offline Rate')
plt.show()

# Normalization

# SPLIT DATA INTO TEST AND TRAIN
np.random.seed(7)

np.random.seed(7)

X = df_work[['hour','week_day','Volume BTC','close']]
Y = df_work[['close']]
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle = False)


# NORMALIZATION
f_transformer = preprocessing.MinMaxScaler((-1,1))
f_transformer = f_transformer.fit(X)

cnt_transformer = preprocessing.MinMaxScaler((-1,1))
cnt_transformer = cnt_transformer.fit(Y)

# no shuffle, time series data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle = False)

X_train_trans = f_transformer.transform(X_train)
X_test_trans = f_transformer.transform(X_test)

y_train_trans = cnt_transformer.transform(y_train)
y_test_trans = cnt_transformer.transform(y_test)

#CREATE LAGGING DATASET FOR TIMESERIES
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 24
# reshape to [samples, time_steps, n_features]
X_train_f, y_train_f = create_dataset(X_train_trans, y_train_trans, time_steps)
X_test_f, y_test_f = create_dataset(X_test_trans, y_test_trans, time_steps)

print("*** SHAPES")
print(X_train_f.shape, y_train_f.shape)
print(X_test_f.shape, y_test_f.shape)


model = keras.Sequential()
model.add(keras.Input(shape=((X_train_f.shape[1], X_train_f.shape[2]))))
#model.add(layers.Bidirectional(layers.LSTM(300, activation = 'tanh', return_sequences=False)))
#model.add(layers.LSTM(300, return_sequences=False, activation = 'tanh'))
model.add(layers.LSTM(300, return_sequences=False, activation = 'tanh'))
model.add(layers.BatchNormalization())
#model.add(layers.Bidirectional(layers.LSTM(120,activation='relu', return_sequences=True)))
#model.add(keras.layers.Dropout(rate=0.2))
#model.add(layers.Flatten())
#model.add(keras.layers.Dense(units=10, activation = 'relu'))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train_f, y_train_f, 
                batch_size = 200, epochs = 50, 
                shuffle=False, validation_split=0.2)

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
plt.figure(figsize=(15,7))
history_df.loc[5:, ['loss', 'val_loss']].plot()
plt.show()

plt.figure(figsize=(15,7))
history_df.loc[5:, ['accuracy', 'val_accuracy']].plot()
plt.show()

print(("Best Validation Loss: {:0.4f}" +\
    "\nBest Validation Accuracy: {:0.4f}")\
    .format(history_df['loss'].min(), 
            history_df['val_accuracy'].max()))