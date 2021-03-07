import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime as datetime
import plotly.express as px

import time
import math 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing
from joblib import dump, load

# Data imported from: http://www.cryptodatadownload.com/data/binance/
df = pd.read_csv('data/Binance_BTCUSDT_d.csv', skiprows=[0])
print(df.dtypes)

print("\n*** DF ***")
df_copy = df[200:].copy()

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
#plt.figure(figsize=(15,7))
#sns.set()
#sns.lineplot(x = df_work.dt, y = 'close',data=df_work).set_title('Offline Rate')
#plt.show()

# Normalization

# SPLIT DATA INTO TEST AND TRAIN
np.random.seed(7)
X = df_work[['hour','week_day','Volume BTC','close']]
Y = df_work[['close']]

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

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=10, # how many epochs to wait before stopping
    restore_best_weights=True,
)

model = keras.Sequential()
model.add(keras.Input(shape=((X_train_f.shape[1], X_train_f.shape[2]))))
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=False)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(256, activation='relu'))
#model.add(layers.LSTM(512, return_sequences=False, activation = 'tanh'))
#model.add(layers.LSTM(300, return_sequences=False, activation = 'tanh'))
#model.add(layers.Flatten())
#model.add(keras.layers.Dense(units=10, activation = 'relu'))
model.add(layers.Dense(units=1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

history = model.fit(X_train_f, y_train_f, 
                batch_size = 50, epochs = 50,
                callbacks=[early_stopping], 
                shuffle=False, validation_split=0.1)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.show()


print(("Best Loss: {:0.4f}" +\
    "\nBest Validation Loss: {:0.4f}")\
    .format(history_df['loss'].min(), 
            history_df['val_loss'].min()))


y_pred = model.predict(X_test_f) 
print(y_pred.shape)
print(y_test_f.shape)

y_test_inv = cnt_transformer.inverse_transform(y_test_f)
y_pred_inv = cnt_transformer.inverse_transform(y_pred)
combined_array = np.concatenate((y_test_inv,y_pred_inv),axis=1)
combined_array2 = np.concatenate((X_test.iloc[time_steps:],combined_array),axis=1)
print(combined_array)

df_final = pd.DataFrame(data = combined_array, columns=["actual", "predicted"])
print("size: %d" % (len(combined_array)))
df_final.head(4)


results = model.evaluate(X_test_f, y_test_f)

#print("Accuracy: %s" % (accuracy_score(y_test_inv, y_pred_inv)))
print("Mean Squared Error: %s" % (mean_squared_error(y_test_inv, y_pred_inv)))
print(results)


# Plots

a = np.repeat(1, len(y_test_inv))
b = np.repeat(2, len(y_pred_inv))

df1 = pd.DataFrame(data = np.concatenate((y_test_inv,(np.reshape(a, (-1, 1)))),axis=1), columns=["price","type"])
df2 = pd.DataFrame(data = np.concatenate((y_pred_inv,(np.reshape(b, (-1, 1)))),axis=1), columns=["price","type"])

frames = [df1, df2]
result = pd.concat(frames, ignore_index=False)

result["type"].replace({1: "actual", 2: "predict"}, inplace=True)
(result[result.type == "actual"]).head(10)

fig = px.line(result, x=result.index.values, y="price", color='type', title='Bitcoin Price')
fig.show()
plt.show()