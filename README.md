# Data-Sufficiency
# 1.Set Up
## 1.1 Enviroments
- Python (Jupyter notebook)
## 1.2 Python requirements
- numpy=1.22.4
- tensorflow=2.10.0
- keras=2.10.0
- matplotlib=3.8.3
- scipy=1.7.3
- scikit-learn=1.3.0
- pandas=2.0.3
# 2. Datasets
# 3. Experiment-Observable data sufficiency(ODS)
 
The entire experiment consists of three steps:
- Design and train the Long Short-Term Memory (LSTM) model in source domain.
- Use limited data in target domain to fine-tune the pre-trained model.
First we
## 3.1 Source domain data processing
We apply LSTM networks to predict battery capacity based on historical feature data. The dataset consists of voltage response and other degradation-related features extracted from battery cycling tests. The goal is to preprocess the data, convert it into a supervised learning format, and train an LSTM model for accurate capacity prediction.

The dataset is read from an Excel file, where battery cycling data for different batteries is stored. Two specific battery types (B26T55 and B1T25) are used, and a set of good features with better transferable capability and predictive capability are selected:
```python
# Read data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path = r'D:\桌面上的文件\数据整理.xlsx'  
df = pd.read_excel(file_path, sheet_name='Exp-32')
# Select battery data
B26T55 = df[df['Battery_Name'] == 'B26T55']
# Selected features for model input
feature_26_T55 = pd.concat([
    B26T55['Vg1'], B26T55['Vg9'], B26T55['RVg'],
    B26T55['Q1'], B26T55['Q2'], B26T55['Q3'], B26T55['Q4'], B26T55['Q5'], B26T55['Q6'], B26T55['Q7'], B26T55['Q9'],
    B26T55['RL1'], B26T55['RL8'], B26T55['RL9'], B26T55['RO8']
], axis=1)
```
To train an LSTM model, we must convert sequential data into a supervised learning format, where past observations (n_in time steps) are used to predict future values (n_out time steps):
```python
def time_series_to_supervised(data, n_in=1, n_out=1,dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    origNames = df.columns
    cols, names = list(), list()
    cols.append(df.shift(0))
    names += [('%s' % origNames[j]) for j in range(n_vars)]
    n_in = max(0, n_in)
    for i in range(n_in, 0, -1):
        time = '(t-%d)' % i
        cols.append(df.shift(i))
        names += [('%s%s' % (origNames[j], time)) for j in range(n_vars)]
    n_out = max(n_out, 0)
    for i in range(1, n_out+1):
        time = '(t+%d)' % i
        cols.append(df.shift(-i))
        names += [('%s%s' % (origNames[j], time)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg
```
Since LSTM networks perform best with normalized data, MinMax scaling is applied to transform feature values between 0 and 1:
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaledFeature_26_T55 = scaler.fit_transform(feature_26_T55)
scaledFeature_26_T55 = pd.DataFrame(data=scaledFeature_26_T55)
```
Converting feature data to time series:
```python
n_steps_in = 3  # Historical time steps
n_steps_out = 1 # Prediction time steps
processedFeature_26_T55 = time_series_to_supervised(scaledFeature_26_T55, n_steps_in, n_steps_out)
```
MinMax scaling is applied to transform capacity values between 0 and 1:
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_26_T55 = scaler.fit_transform(pd.DataFrame(B26T55['Capacity']))
```
Converting capacity data to time series:
```python
processedCapacity_26_T55 = time_series_to_supervised(scaledCapacity_26_T55,n_steps_in,n_steps_out)
```
The processed data is then split into training and testing sets:
```python
# Slice features and labels
data_x26 = processedFeature_26_T55.loc[:,'0(t-3)':'14(t-1)']
data_y26=processedCapacity_26_T55.loc[:,'0']
data_y26=data_y26.values.reshape(-1,1)
# Split training set
train_X26=data_x26.values[:899]
train_y26=data_y26[:899]
train_X26 = train_X26.reshape((train_X26.shape[0], n_steps_in, 15))
```
## 3.2 Train LSTM model
To predict battery capacity, a LSTM network is implemented. LSTM is well-suited for time-series forecasting as it can capture long-term dependencies in sequential data. The LSTM model consists of the following layers: 
- LSTM Layer 1: 96 units, returns sequences (return_sequences=True) to pass outputs to the next LSTM layer.
- LSTM Layer 2: 64 units, does not return sequences (return_sequences=False) as it is the final recurrent layer.
- Dense Layer: 32 neurons with Glorot Uniform initialization.
- Output Layer: 1 neuron for predicting battery capacity.
The model is trained using Mean Squared Error (MSE) as the loss function and the Adam optimizer. The dataset is processed for 75 epochs, meaning it undergoes 75 complete iterations.
Each batch consists of 64 samples, which are processed before updating weights. Shuffling is disabled, ensuring the time-series order is preserved.
```python
# Train Source model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from tensorflow.keras.models import save_model, load_model
import os
import time
# Set random seed
np.random.seed(42)
tf.random.set_seed(42)
# Start time
start_time=time.time()
# Initialize the model
model = Sequential()
model.add(LSTM(96, return_sequences=True, input_shape=(train_X26.shape[1], train_X26.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
# Compile the model
model.compile(loss='mse', optimizer='adam')
print(model.summary())

# Training
history = model.fit(train_X26, train_y26, epochs=75, batch_size=64,verbose=2, shuffle=False)
# End time
end_time = time.time()
# Running time
elapsed_time = end_time - start_time
# Print model running time
print(f"Model running time：{elapsed_time} seconds")
```
## 3.3 Fine-tune the pre-trained LSTM model
As with the source domain data, we first normalize the target domain data and then convert it to time series data:
```python
# Select features
feature_1_T25=pd.concat([B1T25['Vg1'],B1T25['Vg9'],B1T25['RVg'],
                         B1T25['Q1'],B1T25['Q2'],B1T25['Q3'],B1T25['Q4'],B1T25['Q5'],B1T25['Q6'],B1T25['Q7'],B1T25['Q9'],
                         B1T25['RL1'],B1T25['RL8'],B1T25['RL9'],B1T25['RO8']],axis=1)

# Replace the index in front of B1T25 with starting from 0
B1T25.reset_index(drop=True, inplace=True)

# Normalization
scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_1_T25 = scaler.fit_transform(feature_1_T25)
scaledFeature_1_T25 = pd.DataFrame(data=scaledFeature_1_T25)
# Translate T25 features into time series
n_steps_in =3 
n_steps_out=1
processedFeature_1_T25 = time_series_to_supervised(scaledFeature_1_T25,n_steps_in,n_steps_out)
# Normalize the capacity of T25 and convert it into a time series
# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_1_T25 = scaler.fit_transform(pd.DataFrame(B1T25['Capacity']))
processedCapacity_1_T25 = time_series_to_supervised(scaledCapacity_1_T25,n_steps_in,n_steps_out)
# Slice
data_x1 = processedFeature_1_T25.loc[:,'0(t-3)':'14(t-1)']
data_y1=processedCapacity_1_T25.loc[:,'0']
data_y1=data_y1.values.reshape(-1,1)
```
划分训练集和测试集，train_samples之前的数据用来重新训练，之后的数据用来验证。Divide the training dataset and test set train_samples previous data for retraining and subsequent data for validation.

# 4. Experiment-Theoretical data sufficiency(TDS)
# 5. Access
# 6. Acknowledgements
