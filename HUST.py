#!/usr/bin/env python
# coding: utf-8

# In[1]:


# HUST voltage-SOC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle
import matplotlib.cm as cm

with open(r'D:\桌面上的文件\Data Sufficiency\HUST\Data\8-2.pkl', 'rb') as f:
    data = pickle.load(f)

num_cycles = len(data['8-2']['data'])
cmap = cm.get_cmap('Blues', num_cycles)  
colors = cmap(np.linspace(0.3, 1, num_cycles))  

plt.figure(figsize=(4, 3),dpi=600)

for cycle in range(1, num_cycles + 1):
    df = data['8-2']['data'][cycle]

    V1 = df[df['Status'] == 'Constant current charge']['Voltage (V)'].values
    t1 = df[df['Status'] == 'Constant current charge']['Capacity (mAh)'].values
 
    V2 = df[df['Status'] == 'Constant current-constant voltage charge']['Voltage (V)'].values
    t2 = df[df['Status'] == 'Constant current-constant voltage charge']['Capacity (mAh)'].values
   
    time = np.concatenate([t1, t2]) - t1[0]  
    voltage = np.concatenate([V1, V2])

    plt.plot(time, voltage, linewidth=1, alpha=0.6,color=colors[cycle - 1])

plt.title('Voltage vs Capacity for All Cycles')
plt.xlabel('Charging Capacity (mAh)')
plt.ylabel('Charging Voltage (V)')
plt.ylim(2,4.25)
plt.show()


# In[2]:


#HUST capacity-cycle
import pandas as pd
import matplotlib.pyplot as plt
import os

file1 = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_42_filtered.xlsx'
file2 = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_82_filtered.xlsx'

data1 = pd.read_excel(file1, header=None)  
capacity1 = data1.iloc[1:,9]  

data2 = pd.read_excel(file2, header=None)  
capacity2 = data2.iloc[1:, 9] 

plt.figure(figsize=(4, 3), dpi=600)
plt.plot(capacity1, label='File 2', marker='o', markersize=3, color=plt.cm.Reds(0.8))
plt.plot(capacity2, label='File 1', marker='o', markersize=3, color=plt.cm.Blues(0.8))

plt.title('HUST', fontsize=10)
plt.xlabel('Cycle Index', fontsize=10)
plt.ylabel('Capacity (Ah)', fontsize=10)

plt.tight_layout()
plt.show()


# In[ ]:


# Feature extraction +savgol_filter 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter

f = open(r'D:\桌面上的文件\Data Sufficiency\HUST\8-2.pkl','rb')#change file name to extract different battery features
data = pickle.load(f)

results = []

for cycle in range(len(data['8-2']['data'])):
    df = data['8-2']['data'][cycle+1]
    
    V1 = df[df['Status'] == 'Constant current charge']['Voltage (V)'][10:]
    t1 = df[df['Status'] == 'Constant current charge']['Time (s)'][10:]
    
    V2 = df[df['Status'] == 'Constant current-constant voltage charge']['Voltage (V)']

    voltage_diff = V2.diff().fillna(0)
    
    threshold = 0.0005
    increasing_indices = voltage_diff[voltage_diff > threshold].index

    increasing_voltage = V2[increasing_indices]
    increasing_time = df[df['Status'] == 'Constant current-constant voltage charge']['Time (s)'][increasing_indices]
    
    Gra_1 = np.diff(V1) / np.diff(t1)
    Vg1 = np.mean(Gra_1)
    
    Gra_2 = np.diff(increasing_voltage) / np.diff(increasing_time)
    Vg2 = np.mean(Gra_2)
    
    RL1 = (df[df['Status'] == 'Constant current charge']['Voltage (V)'].iloc[-1] - 
           df[df['Status'] == 'Constant current charge']['Voltage (V)'].iloc[0]) / 5.5
    RL2 = (V2.iloc[-1] - V2.iloc[0]) / 1.1
    RO = (V1.iloc[-1] - V2.iloc[0]) / (5.5 - 1.1)  
    
    Q1 = df[df['Status'] == 'Constant current charge']['Capacity (mAh)'].iloc[-1]
    Q2 = df[df['Status'] == 'Constant current-constant voltage charge']['Capacity (mAh)'][increasing_indices].iloc[-1]
    
    tVD2 = increasing_time.iloc[-1]
    cap = df[df['Status'] == 'Constant current discharge_0']['Capacity (mAh)'].iloc[0]

    cycle_features = {
        'Cycle': cycle+1,
        'Vg1': Vg1,
        'Vg2': Vg2,
        'Q1': Q1,
        'Q2': Q2,
        'RL1': RL1,
        'RL2': RL2,
        'RO': RO,
        'tVD2': tVD2,
        'Capacity': cap
    }

    results.append(cycle_features)

results_df = pd.DataFrame(results)

window_length = 11  
polyorder = 2  

for col in results_df.columns:
    if col != 'Cycle' and col != 'Capacity':
        results_df[col] = savgol_filter(results_df[col], window_length, polyorder)

print(results_df)

results_df.to_excel('D:\桌面上的文件\Features_HUST_82_filtered.xlsx', index=False)

print("特征提取和滤波完成，并已保存")


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_82_filtered.xlsx'
df = pd.read_excel(file_path).iloc[5:,]

columns_to_analyze = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']

def calculate_correlations(df, cycles, target='Capacity'):
    cycle_data = df[df['Cycle'] <= cycles]
    correlations = cycle_data[columns_to_analyze + [target]].corr()[target].drop(target)
    return abs(correlations)

correlations = {}
for cycle in range(20, len(df)+1,20):
    correlations[cycle] = calculate_correlations(df, cycle)

average_correlations = np.mean([list(correlations[cycle].values) for cycle in range(20, len(df)+1,20)], axis=1)

plt.figure(figsize=(10, 6),dpi=600)

for feature in columns_to_analyze:
    corr_values = [correlations[cycle][feature] for cycle in range(20, len(df)+1,20)]
    plt.plot(range(20, len(df)+1,20), corr_values, marker='o', linestyle='-', label=feature)

plt.plot(range(20, len(df)+1,20), average_correlations, marker='o', linestyle='-', color='black', label='Average')

plt.title('Correlations with Capacity 8-2')
plt.xlabel('Cycles')
plt.ylabel('Correlation')
plt.xticks(range(100, len(df) + 1, 100))
plt.legend()
plt.tight_layout()
plt.show()


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_42_filtered.xlsx'
df = pd.read_excel(file_path).iloc[5:,]

columns_to_analyze = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']

def calculate_correlations(df, cycles, target='Capacity'):
    cycle_data = df[df['Cycle'] <= cycles]
    correlations = cycle_data[columns_to_analyze + [target]].corr()[target].drop(target)
    return abs(correlations)

correlations_42 = {}
for cycle in range(20, len(df)+1,20):
    correlations_42[cycle] = calculate_correlations(df, cycle)

average_correlations = np.mean([list(correlations_42[cycle].values) for cycle in range(20, len(df)+1,20)], axis=1)

plt.figure(figsize=(10, 6),dpi=600)

for feature in columns_to_analyze:
    corr_values = [correlations_42[cycle][feature] for cycle in range(20, len(df)+1,20)]
    plt.plot(range(20, len(df)+1,20), corr_values, marker='o', linestyle='-', label=feature)

plt.plot(range(20, len(df)+1,20), average_correlations, marker='o', linestyle='-', color='black', label='Average')

plt.title('Correlations with Capacity 4-2')
plt.xlabel('Cycles')
plt.ylabel('Correlation')
plt.xticks(range(100, len(df) + 1, 100))
plt.legend()
plt.tight_layout()
plt.show()


# In[7]:


import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

file_path_1 = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_42_filtered.xlsx'
file_path_2 = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_82_filtered.xlsx'

df1 = pd.read_excel(file_path_1).iloc[5:1700, :]
df2 = pd.read_excel(file_path_2).iloc[5:1700, :]

common_columns = list(set(df1.columns) & set(df2.columns))
columns_to_analyze=[ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

df1[columns_to_analyze] = scaler1.fit_transform(df1[columns_to_analyze])
df2[columns_to_analyze] = scaler2.fit_transform(df2[columns_to_analyze])

def calculate_wasserstein_distance(df1, df2, cycles, columns):
    distances = {}
    for col in columns:
        data1 = df1.iloc[:cycles][col].dropna()
        data2 = df2.iloc[:cycles][col].dropna()
        if not data1.empty and not data2.empty:
            distance = wasserstein_distance(data1, data2)
            distances[col] = 1 - distance
    return distances

max_cycles = min(df1['Cycle'].max(), df2['Cycle'].max())
cycle_ranges = range(20, max_cycles + 1, 20)

wasserstein_distances = {col: [] for col in columns_to_analyze}
average_distances = []

for cycle_range in cycle_ranges:
    distances = calculate_wasserstein_distance(df1, df2, cycle_range, columns_to_analyze)
    for col, distance in distances.items():
        wasserstein_distances[col].append(distance)

    if distances:
        average_distance = np.mean(list(distances.values()))
        average_distances.append(average_distance)
    else:
        average_distances.append(0)

plt.figure(figsize=(10, 6),dpi=600)

for col, distances in wasserstein_distances.items():
    plt.plot(cycle_ranges, distances, marker='o', linestyle='-', label=col)

plt.plot(cycle_ranges, average_distances, marker='o', linestyle='--', color='black', label='Average')
plt.ylim()
plt.title('1-Wasserstein Distance between Features')
plt.xlabel('Cycles')
plt.ylabel('1-Wasserstein Distance')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


#HUST PC TC excel
import pandas as pd
import numpy as np

df_wasserstein = pd.DataFrame(wasserstein_distances)
correlations_df = pd.DataFrame(correlations)
correlations_df_42 = pd.DataFrame(correlations_42)

correlations_df = correlations_df.T
df_correlation = pd.DataFrame(correlations_df)
correlations_df_42 = correlations_df_42.T
df_correlation_42 = pd.DataFrame(correlations_df_42)

file_path = 'D:\\桌面上的文件\\HUST.xlsx'
with pd.ExcelWriter(file_path) as writer:
    df_wasserstein.to_excel(writer, sheet_name='TC_Trend', index=False)
    df_correlation.to_excel(writer, sheet_name='PC_Trend', index=False)
    df_correlation_42.to_excel(writer, sheet_name='PC_Trend_42', index=False)

print(f"数据已成功写入 {file_path}")


# In[6]:


#PC TC Trend
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\Data Sufficiency\HUST\HUST.xlsx'  
df_TC = pd.read_excel(file_path, sheet_name='TC_Trend')  
df_PC = pd.read_excel(file_path, sheet_name='PC_Trend')  

mean_wasserstein = df_TC.mean(axis=1)
mean_correlation = df_PC.mean(axis=1)

fig, ax1 = plt.subplots(figsize=(4, 2), dpi=600)

color = '#599CB4'  
ax1.set_xlabel('Periods')
ax1.set_ylabel('Average Wasserstein Trend', color=color)
ax1.plot(mean_wasserstein[:20], marker='o', markersize=7,linestyle='-', linewidth=4,color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.7, 1.0)  

ax2 = ax1.twinx() 
color = '#C25759'  
ax2.set_ylabel('Average Correlation Trend', color=color)
ax2.plot(mean_correlation[:20], marker='o', markersize=10,linestyle='-', linewidth=5,color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.2, 1)  
plt.show()


# In[ ]:





# In[2]:


#LSTM
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def time_series_to_supervised(data, n_in=1, n_out=1,dropnan=True):
    """
    :param data:作为列表或2D NumPy数组的观察序列。需要。
    :param n_in:作为输入的滞后观察数（X）。值可以在[1..len（数据）]之间可选。默认为1。
    :param n_out:作为输出的观测数量（y）。值可以在[0..len（数据）]之间。可选的。默认为1。
    :param dropnan:Boolean是否删除具有NaN值的行。可选的。默认为True。
    :return:
    """
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

from sklearn.decomposition import PCA
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path_5 = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_42_filtered.xlsx'
selected_columns = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[5:1706]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_5 = scaler.fit_transform(Feature_5)

scaledFeature_5 = pd.DataFrame(data=scaledFeature_5)

print(scaledFeature_5.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_5 = time_series_to_supervised(scaledFeature_5,n_steps_in,n_steps_out)
scaler = MinMaxScaler(feature_range=(0, 1))
cap=(pd.read_excel(file_path_5).iloc[5:1706,-1]).values.reshape(-1, 1)
scaledCapacity_5 = scaler.fit_transform(cap)
n_steps_in =3 
n_steps_out=1
processedCapacity_5 = time_series_to_supervised(scaledCapacity_5,n_steps_in,n_steps_out)

data_x5 = processedFeature_5.loc[:,'0(t-3)':'6(t-1)']
data_y5=processedCapacity_5.loc[:,'0']
train_X5=data_x5.values[:1706]
train_y5=data_y5[:1706]
train_X5 = train_X5.reshape((train_X5.shape[0], n_steps_in, 7))


# In[3]:


#Train source model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from tensorflow.keras.models import save_model, load_model
import os
import time

np.random.seed(42)
tf.random.set_seed(42)

start_time=time.time()

model = Sequential()
model.add(LSTM(96, return_sequences=True, input_shape=(train_X5.shape[1], train_X5.shape[2]))) 

model.add(LSTM(64, return_sequences=False))

model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

history = model.fit(train_X5, train_y5, epochs=30, batch_size=64,verbose=2, shuffle=False)

end_time = time.time()
elapsed_time = end_time - start_time
# 打印模型运行时间
#print(f"模型运行时间：{elapsed_time} 秒")


# In[ ]:


# Automatically look at different fine-tuning cycles
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from keras.models import Model
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tensorflow.keras.models import save_model, load_model
import os
import time


def time_series_to_supervised(data, n_in=1, n_out=1,dropnan=True):
    """
    :param data:作为列表或2D NumPy数组的观察序列。需要。
    :param n_in:作为输入的滞后观察数（X）。值可以在[1..len（数据）]之间可选。默认为1。
    :param n_out:作为输出的观测数量（y）。值可以在[0..len（数据）]之间。可选的。默认为1。
    :param dropnan:Boolean是否删除具有NaN值的行。可选的。默认为True。
    :return:
    """
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

file_path_5 = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_42_filtered.xlsx'  
selected_columns = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[5:1706]
file_path_6 = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_82_filtered.xlsx'  
Feature_6 = pd.read_excel(file_path_6, usecols=selected_columns).iloc[5:2283]

capacity_5=pd.read_excel(file_path_5).iloc[5:1706,-1]
capacity_6=pd.read_excel(file_path_6).iloc[5:2283,-1]
cap_6=(capacity_6).values.reshape(-1, 1)
n_steps_in=3
n_steps_out = 1

np.random.seed(42)
tf.random.set_seed(42)

prediction_capability=[]
for train_samples in range(20,401,20):

    scaler = MinMaxScaler(feature_range=(0,1))
    scaledFeature_6 = scaler.fit_transform(Feature_6)
   
    scaledFeature_6 = pd.DataFrame(data=scaledFeature_6)
    
    n_steps_in =3 
    n_steps_out=1
    processedFeature_6 = time_series_to_supervised(scaledFeature_6,n_steps_in,n_steps_out)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledCapacity_6 = scaler.fit_transform(cap_6)
   
    n_steps_in =3 
    n_steps_out=1
    processedCapacity_6 = time_series_to_supervised(scaledCapacity_6,n_steps_in,n_steps_out)
   
    data_x6 = processedFeature_6.loc[:,'0(t-3)':'6(t-1)']
    data_y6=processedCapacity_6.loc[:,'0']
    data_y6=data_y6.values.reshape(-1,1)
   
    train_X6=data_x6.values[:train_samples]
    test_X6=data_x6.values[train_samples:]
    train_y6=data_y6[:train_samples]
    test_y6=data_y6[train_samples:]
    train_X6 = train_X6.reshape((train_X6.shape[0], n_steps_in, 7))
    test_X6 = test_X6.reshape((test_X6.shape[0], n_steps_in, 7))
    
    for layer in model.layers[:2]:  
        layer.trainable = False

    input_layer = Input(shape=(train_X6.shape[1], train_X6.shape[2]))
    lstm_output_1 = model.layers[0](input_layer) 
    lstm_output_2 = model.layers[1](lstm_output_1)  
    new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)
    new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)
    
    transfer_model = Model(inputs=input_layer, outputs=new_output_layer)
    transfer_model.compile(loss='mse', optimizer='adam')
    transfer_model.fit(train_X6, train_y6, epochs=50, batch_size=64, verbose=2, shuffle=False)

    yhat6t= transfer_model.predict(test_X6)
    test_y6=test_y6.reshape(-1,1) 
    inv_forecast_y6t = scaler.inverse_transform(yhat6t)
    inv_test_y6t = scaler.inverse_transform(test_y6)
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true))
    mape_6t = mape(inv_test_y6t, inv_forecast_y6t)
    print('Test MAPE: %.3f' % mape_6t)
    prediction_capability.append(1-mape_6t)
prediction_capability


# In[ ]:


# HUST Acc
import pandas as pd

prediction_df = pd.DataFrame(prediction_capability, columns=["42 to 82"])

file_path_output = "D:\桌面上的文件\HUST_Acc.xlsx"

prediction_df.to_excel(file_path_output, index=False)


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

file_path = 'D:\桌面上的文件\Data Sufficiency\HUST\HUST_Acc.xlsx'  
df = pd.read_excel(file_path)

column_data = df['42 to 82']
column_data = df['42 to 82']

plt.figure(figsize=(5, 2), dpi=600)

for i in range(len(column_data)):
    if i == 2:  # 第四个柱子
        plt.bar(i, column_data[i], width=0.8, color='#898988')
    else:
        plt.bar(i, column_data[i], width=0.8, color='#d7d7d7')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.grid(False)
plt.ylim(0.97,0.998)
plt.show()


# In[1]:


import pandas as pd

# Step 1: 读取Excel文件
file_path = 'D:\桌面上的文件\HUST.xlsx'
tc_data = pd.read_excel(file_path, sheet_name='TC_Trend')  # f(t)
pc_data = pd.read_excel(file_path, sheet_name='PC_Trend').iloc[:85]  # g(t)

# Step 2: 计算每个周期的TC和PC的平均值
tc_avg = tc_data.mean(axis=1)  # 计算每一行的均值，得到 f(t) 的均值
pc_avg = pc_data.mean(axis=1)  # 计算每一行的均值，得到 g(t) 的均值
t_values = tc_data.index + 1  # 周期 t，从1开始

# Step 3: 计算 f(t) 和 g(t) 的最大值
max_f = tc_avg.max()  # TC平均值的最大值
max_g = pc_avg.max()  # PC平均值的最大值
max_t = t_values.max()

# Step 4: 归一化 f(t) 和 g(t)
tc_normalized = tc_avg / max_f  # 归一化 f(t)
pc_normalized = pc_avg / max_g  # 归一化 g(t)
t_normalized = pc_avg / max_g  # 归一化 g(t)

# Step 5: 计算目标函数 (f(t)/max_f + g(t)/max_g)/t
t_values = tc_data.index + 1  # 周期 t，从1开始
scores = (tc_normalized + pc_normalized) / (t_values/max_t)

# Step 6: 找到最大值的位置
max_score_idx = scores[:20].idxmax()  # 获取最大值的索引
max_score_value = scores[:20].max()   # 最大值的分数

# 输出结果
print(f"最大值的周期 t：{max_score_idx + 1}")  # +1 是因为索引从0开始
print(f"最大值的分数：{max_score_value}")


# In[13]:


scores


# In[2]:


import pandas as pd

# Step 1: 读取Excel文件
file_path = 'D:\桌面上的文件\HUST.xlsx'
tc_data = pd.read_excel(file_path, sheet_name='TC_Trend')  # f(t)
pc_data = pd.read_excel(file_path, sheet_name='PC_Trend').iloc[:85]  # g(t)

# Step 2: 计算每个周期的TC和PC的平均值
tc_avg = tc_data.mean(axis=1)  # 计算每一行的均值，得到 f(t) 的均值
pc_avg = pc_data.mean(axis=1)  # 计算每一行的均值，得到 g(t) 的均值

# Step 3: 计算 f(t) 和 g(t) 的最大值
max_f = tc_avg.max()  # TC平均值的最大值
max_g = pc_avg.max()  # PC平均值的最大值

# Step 4: 归一化 f(t) 和 g(t)
tc_normalized = tc_avg / max_f  # 归一化 f(t)
pc_normalized = pc_avg / max_g  # 归一化 g(t)

# Step 5: 计算目标函数 (f(t)/max_f + g(t)/max_g)/t
t_values = tc_data.index + 1  # 周期 t，从1开始
scores = (tc_normalized * pc_normalized) / t_values

# Step 6: 找到最大值的位置
max_score_idx = scores.idxmax()  # 获取最大值的索引
max_score_value = scores.max()   # 最大值的分数

# 输出结果
print(f"最大值的周期 t：{max_score_idx + 1}")  # +1 是因为索引从0开始
print(f"最大值的分数：{max_score_value}")


# In[5]:


import pandas as pd

# Step 1: 读取Excel文件
file_path = 'D:\桌面上的文件\HUST.xlsx'
tc_data = pd.read_excel(file_path, sheet_name='TC_Trend')  # f(t)
pc_data = pd.read_excel(file_path, sheet_name='PC_Trend').iloc[:85]  # g(t)

# Step 2: 计算每个周期的TC和PC的平均值
tc_avg = tc_data.mean(axis=1)  # 计算每一行的均值，得到 f(t) 的均值
pc_avg = pc_data.mean(axis=1)  # 计算每一行的均值，得到 g(t) 的均值

# Step 3: 计算 f(t) 和 g(t) 的最大值
max_f = tc_avg.max()  # TC平均值的最大值
max_g = pc_avg.max()  # PC平均值的最大值

# Step 4: 归一化 f(t) 和 g(t)
tc_normalized = tc_avg / max_f  # 归一化 f(t)
pc_normalized = pc_avg / max_g  # 归一化 g(t)

# Step 5: 计算目标函数 (f(t)/max_f + g(t)/max_g)/t
t_values = tc_data.index + 1  # 周期 t，从1开始
scores = (tc_normalized + pc_normalized) 

# Step 6: 找到最大值的位置
max_score_idx = scores[:20].idxmax()  # 获取最大值的索引
max_score_value = scores[:20].max()   # 最大值的分数

# 输出结果
print(f"最大值的周期 t：{max_score_idx + 1}")  # +1 是因为索引从0开始
print(f"最大值的分数：{max_score_value}")


# In[10]:


#所有电池的特征
import os
import numpy as np
import pandas as pd
import pickle
from scipy.signal import savgol_filter

# 定义文件夹路径
folder_path = r'D:\桌面上的文件\Data Sufficiency\HUST\Data'

# 获取文件夹中所有.pkl文件的路径
pkl_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.pkl')]

all_results = []  # 用于存储所有文件的结果

# 遍历所有.pkl文件
for file_index, file_path in enumerate(pkl_files, start=1):
    print(f"正在处理第 {file_index}/{len(pkl_files)} 个文件: {os.path.basename(file_path)}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 遍历每个电池（假设电池标识为文件内部的一级键）
    for battery_id in data.keys():
        battery_data = data[battery_id]['data']
        
        # 遍历每个周期的数据
        for cycle in range(len(battery_data)):
            print(f"  - 正在处理电池 {battery_id} 的第 {cycle + 1}/{len(battery_data)} 个周期")
            df = battery_data[cycle + 1]  # 周期数据
            
            V1 = df[df['Status'] == 'Constant current charge']['Voltage (V)'][10:]
            t1 = df[df['Status'] == 'Constant current charge']['Time (s)'][10:]
            
            V2 = df[df['Status'] == 'Constant current-constant voltage charge']['Voltage (V)']
            voltage_diff = V2.diff().fillna(0)
            threshold = 0.0005
            increasing_indices = voltage_diff[voltage_diff > threshold].index
            
            increasing_voltage = V2[increasing_indices]
            increasing_time = df[df['Status'] == 'Constant current-constant voltage charge']['Time (s)'][increasing_indices]
            
            Gra_1 = np.diff(V1) / np.diff(t1)
            Vg1 = np.mean(Gra_1)
            
            Gra_2 = np.diff(increasing_voltage) / np.diff(increasing_time)
            Vg2 = np.mean(Gra_2)
            
            RL1 = (df[df['Status'] == 'Constant current charge']['Voltage (V)'].iloc[-1] - 
                   df[df['Status'] == 'Constant current charge']['Voltage (V)'].iloc[0]) / 5.5
            RL2 = (V2.iloc[-1] - V2.iloc[0]) / 1.1
            RO = (V1.iloc[-1] - V2.iloc[0]) / (5.5 - 1.1)
            
            Q1 = df[df['Status'] == 'Constant current charge']['Capacity (mAh)'].iloc[-1]
            Q2 = df[df['Status'] == 'Constant current-constant voltage charge']['Capacity (mAh)'][increasing_indices].iloc[-1]
            
            tVD2 = increasing_time.iloc[-1]
            cap = df[df['Status'] == 'Constant current discharge_0']['Capacity (mAh)'].iloc[0]
            
            # 保存每个周期的特征
            cycle_features = {
                'Battery': battery_id,
                'Cycle': cycle + 1,
                'Vg1': Vg1,
                'Vg2': Vg2,
                'Q1': Q1,
                'Q2': Q2,
                'RL1': RL1,
                'RL2': RL2,
                'RO': RO,
                'tVD2': tVD2,
                'Capacity': cap
            }
            
            all_results.append(cycle_features)

# 转换为数据框
results_df = pd.DataFrame(all_results)

# 定义 Savitzky-Golay 滤波器参数
window_length = 11  # 窗口长度（必须为奇数）
polyorder = 2  # 多项式阶数

# 对除 Battery、Cycle 和 Capacity 以外的特征进行滤波
for col in results_df.columns:
    if col not in ['Battery', 'Cycle', 'Capacity']:
        results_df[col] = savgol_filter(results_df[col], window_length, polyorder)

# 保存结果到Excel文件
output_path = r'D:\桌面上的文件\所有特征提取结果.xlsx'
results_df.to_excel(output_path, index=False)

print("特征提取和滤波完成，并已保存到", output_path)


# In[7]:


#实验25
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def time_series_to_supervised(data, n_in=1, n_out=1,dropnan=True):
    """
    :param data:作为列表或2D NumPy数组的观察序列。需要。
    :param n_in:作为输入的滞后观察数（X）。值可以在[1..len（数据）]之间可选。默认为1。
    :param n_out:作为输出的观测数量（y）。值可以在[0..len（数据）]之间。可选的。默认为1。
    :param dropnan:Boolean是否删除具有NaN值的行。可选的。默认为True。
    :return:
    """
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
#先归一化再PCA
from sklearn.decomposition import PCA
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# 读取 Excel 文件中的数据
file_path = r'D:\桌面上的文件\DL-HUST特征和标签.xlsx'  
df = pd.read_excel(file_path)

# 电池类型列表
battery_types = ['B26T55', 'B1T25']
feature_names=['Vg1','Vg2','Q1','Q2','RL1','RL2','RO','tVD2','Capacity']
B26T55 = df[df['Battery'] == '1-1']
B1T25 = df[df['Battery'] == '10-8']
# #全特征
# feature_26_T55=pd.concat([B26T55['VC89'],B26T55['VD89'],B26T55['tVD89'],B26T55['ReVC'],B26T55['ReVD'],B26T55['tReVD'],
#                           B26T55['Vg1'],B26T55['Vg2'],B26T55['Vg3'],B26T55['Vg4'],B26T55['Vg5'],B26T55['Vg6'],B26T55['Vg7'],B26T55['Vg8'],B26T55['Vg9'],B26T55['RVg'],
#                           B26T55['Q1'],B26T55['Q2'],B26T55['Q3'],B26T55['Q4'],B26T55['Q5'],B26T55['Q6'],B26T55['Q7'],B26T55['Q8'],B26T55['Q9'],
#                           B26T55['RL1'],B26T55['RL2'],B26T55['RL3'],B26T55['RL4'],B26T55['RL5'],B26T55['RL6'],B26T55['RL7'],B26T55['RL8'],B26T55['RL9'],
#                           B26T55['RO1'],B26T55['RO2'],B26T55['RO3'],B26T55['RO4'],B26T55['RO5'],B26T55['RO6'],B26T55['RO7'],B26T55['RO8']],axis=1)
#最好的特征
feature_26_T55=pd.concat([B26T55['Vg1'],B26T55['Vg2'],B26T55['Q1'],
                          B26T55['Q2'],B26T55['RL1'],B26T55['RL2'],B26T55['RO'],B26T55['tVD2']],axis=1)
# #除去迁移能力的特征
# feature_26_T55=pd.concat([B26T55['Vg1'],B26T55['Vg9'],
#                           B26T55['Q1'],B26T55['Q2'],B26T55['Q3'],B26T55['Q4'],B26T55['Q5'],B26T55['Q6'],B26T55['Q7'],B26T55['Q9'],
#                           B26T55['RL1'],B26T55['RL9'],
#                           B26T55['RO8']],axis=1)
# #选不同特征组 Q组
# feature_26_T55=pd.concat([B26T55['Q9'],B26T55['Q8'],B26T55['Q7'],B26T55['Q6'],B26T55['Q5'],B26T55['Q4'],B26T55['Q3'],B26T55['Q2'],B26T55['Q1']],axis=1)
# #选不同特征组 Vg组
# feature_26_T55=pd.concat([B26T55['RVg'],B26T55['Vg9'],B26T55['Vg8'],B26T55['Vg7'],B26T55['Vg6'],B26T55['Vg5'],B26T55['Vg4'],B26T55['Vg3'],B26T55['Vg2'],B26T55['Vg1']],axis=1)
# #选不同特征组 RL组
# feature_26_T55=pd.concat([B26T55['RL9'],B26T55['RL8'],B26T55['RL7'],B26T55['RL6'],B26T55['RL5'],B26T55['RL4'],B26T55['RL3'],B26T55['RL2'],B26T55['RL1']],axis=1)
# #选不同特征组 RO组
# feature_26_T55=pd.concat([B26T55['RO8'],B26T55['RO7'],B26T55['RO6'],B26T55['RO5'],B26T55['RO4'],B26T55['RO3'],B26T55['RO2'],B26T55['RO1']],axis=1)
#选不同特征组 VCVD组
#feature_26_T55=pd.concat([B26T55['VC89'],B26T55['VD89'],B26T55['tVD89'],B26T55['ReVC'],B26T55['ReVD'],B26T55['tReVD']],axis=1)
#首先归一化，再移位处理，得到训练样本
scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_26_T55 = scaler.fit_transform(feature_26_T55)
# # 使用 PCA 进行降维
# pca = PCA(n_components=4)
# scaledFeature_26_T55 = pca.fit_transform(scaledFeature_26_T55)

# 创建包含降维结果的新 DataFrame
scaledFeature_26_T55 = pd.DataFrame(data=scaledFeature_26_T55)

print(scaledFeature_26_T55.shape)

#实验25
#把T55的特征转化为时间序列
n_steps_in =3 #历史时间长度
n_steps_out=1#预测时间长度
processedFeature_26_T55 = time_series_to_supervised(scaledFeature_26_T55,n_steps_in,n_steps_out)
#print(processedFeature_26_T55.head())
#把T55的容量归一化并转化为时间序列
#首先归一化，再移位处理，得到训练样本
scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_26_T55 = scaler.fit_transform(pd.DataFrame(B26T55['Capacity']))
#print(scaledCapacity_26_T55.shape)

n_steps_in =3 #历史时间长度
n_steps_out=1#预测时间长度
processedCapacity_26_T55 = time_series_to_supervised(scaledCapacity_26_T55,n_steps_in,n_steps_out)
#print(processedCapacity_1_T25.head())
#切片
data_x26 = processedFeature_26_T55.loc[:,'0(t-3)':'7(t-1)']
data_y26=processedCapacity_26_T55.loc[:,'0']
data_y26=data_y26.values.reshape(-1,1)
#划分训练集和测试集
train_X26=data_x26.values[:1500]#注意dataframe用reshape,要先用values
#test_X26=data_x26.values[:]
train_y26=data_y26[:1500]
#test_y26=data_y26[:]
train_X26 = train_X26.reshape((train_X26.shape[0], n_steps_in, 8))
#test_X26 = test_X26.reshape((test_X26.shape[0], n_steps_in, 16))
#print(train_X26.shape, train_y26.shape, test_X26.shape, test_y26.shape)


# In[8]:


#实验24
#把T55的特征转化为时间序列
n_steps_in =3 #历史时间长度
n_steps_out=1#预测时间长度
processedFeature_26_T55 = time_series_to_supervised(scaledFeature_26_T55,n_steps_in,n_steps_out)
#print(processedFeature_26_T55.head())
#把T55的容量归一化并转化为时间序列
#首先归一化，再移位处理，得到训练样本
scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_26_T55 = scaler.fit_transform(pd.DataFrame(B26T55['Capacity']))
#print(scaledCapacity_26_T55.shape)
soure_data=899

n_steps_in =3 #历史时间长度
n_steps_out=1#预测时间长度
processedCapacity_26_T55 = time_series_to_supervised(scaledCapacity_26_T55,n_steps_in,n_steps_out)
#print(processedCapacity_1_T25.head())
#切片
data_x26 = processedFeature_26_T55.loc[:,'0(t-3)':'7(t-1)']#过去三步的特征
data_y26=processedCapacity_26_T55.loc[:,'0']#当前的容量
data_y26=data_y26.values.reshape(-1,1)#转化为二维数组
#划分训练集和测试集
train_X26=data_x26.values[:soure_data]#注意dataframe用reshape,要先用values
#test_X26=data_x26.values[:]
train_y26=data_y26[:soure_data]
#test_y26=data_y26[:]
train_X26 = train_X26.reshape((train_X26.shape[0], n_steps_in,8))#转为三维数组，过去3个时间步的16个特征。历史时间步的特征作为输入，当前步的容量作为输出


# In[9]:


#实验24
#训练LSTM模型，特征是4维，模型叫做model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from tensorflow.keras.models import save_model, load_model
import os
import time
# 设置全局随机种子
np.random.seed(42)
tf.random.set_seed(42)
#记录开始时间
start_time=time.time()
# 构建网络
model = Sequential()#序列类型神经网络
model.add(LSTM(96, return_sequences=True, input_shape=(train_X26.shape[1], train_X26.shape[2]))) # 返回完整的输出序列，input_shape定义输出数据的形状
model.add(LSTM(64, return_sequences=False))#只返回输出序列的最后一步

model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

# 训练
#history = model.fit(train_X26, train_y26, epochs=30, batch_size=64, validation_data=(test_X26, test_y26), verbose=2, shuffle=False)
history = model.fit(train_X26, train_y26, epochs=30, batch_size=64,verbose=2, shuffle=False)#epoch训练的轮数，batch_size每个批次的样本数量
# 记录模型结束运行的时间
end_time = time.time()
# 计算模型运行时间
elapsed_time = end_time - start_time
# 打印模型运行时间
print(f"模型运行时间：{elapsed_time} 秒")

# # plot history
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()
# 查看训练好的模型中的权重
#print('训练完成后权重分配为：%s\n' % (model.layers[0].get_weights()))


# In[10]:


#实验24 55到25
train_samples=300
import matplotlib
#最好的特征
feature_1_T25=pd.concat([B1T25['Vg1'],B1T25['Vg2'],B1T25['Q1'],
                         B1T25['Q2'],B1T25['RL1'],B1T25['RL2'],B1T25['RO'],B1T25['tVD2']],axis=1)
# #全部特征
# feature_1_T25=pd.concat([B1T25['VC89'],B1T25['VD89'],B1T25['tVD89'],B1T25['ReVC'],B1T25['ReVD'],B1T25['tReVD'],
#                          B1T25['Vg1'],B1T25['Vg2'],B1T25['Vg3'],B1T25['Vg4'],B1T25['Vg5'],B1T25['Vg6'],B1T25['Vg7'],B1T25['Vg8'],B1T25['Vg9'],B1T25['RVg'],
#                          B1T25['Q1'],B1T25['Q2'],B1T25['Q3'],B1T25['Q4'],B1T25['Q5'],B1T25['Q6'],B1T25['Q7'],B1T25['Q8'],B1T25['Q9'],
#                          B1T25['RL1'],B1T25['RL2'],B1T25['RL3'],B1T25['RL4'],B1T25['RL5'],B1T25['RL6'],B1T25['RL7'],B1T25['RL8'],B1T25['RL9'],
#                          B1T25['RO1'],B1T25['RO2'],B1T25['RO3'],B1T25['RO4'],B1T25['RO5'],B1T25['RO6'],B1T25['RO7'],B1T25['RO8']
#                         ],axis=1)
# #不加迁移能力高的特征
# feature_1_T25=pd.concat([B1T25['Vg1'],B1T25['Vg9'],
#                          B1T25['Q1'],B1T25['Q2'],B1T25['Q3'],B1T25['Q4'],B1T25['Q5'],B1T25['Q6'],B1T25['Q7'],B1T25['Q9'],
#                          B1T25['RL1'],B1T25['RL9'],B1T25['RO8']],axis=1)
# #选择分组特征
# feature_1_T25=pd.concat([
#                          B1T25['Q1'],B1T25['Q2'],B1T25['Q3'],B1T25['Q4'],B1T25['Q5'],B1T25['Q6'],B1T25['Q7'],B1T25['Q8'],B1T25['Q9'],
#                          ],axis=1)
# #选择分组特征Vg组
# feature_1_T25=pd.concat([
#                          B1T25['Vg1'],B1T25['Vg2'],B1T25['Vg3'],B1T25['Vg4'],B1T25['Vg5'],B1T25['Vg6'],B1T25['Vg7'],B1T25['Vg8'],B1T25['Vg9'],B1T25['RVg']
#                          ],axis=1)
# #选择分组特征RL组
# feature_1_T25=pd.concat([
#                          B1T25['RL1'],B1T25['RL2'],B1T25['RL3'],B1T25['RL4'],B1T25['RL5'],B1T25['RL6'],B1T25['RL7'],B1T25['RL8'],B1T25['RL9']
#                          ],axis=1)
# #选择分组特征RO组
# feature_1_T25=pd.concat([
#                          B1T25['RO1'],B1T25['RO2'],B1T25['RO3'],B1T25['RO4'],B1T25['RO5'],B1T25['RO6'],B1T25['RO7'],B1T25['RO8']
#                          ],axis=1)
#选择分组特征VCVD组
# feature_1_T25=pd.concat([
#                          B1T25['VC89'],B1T25['VD89'],B1T25['tVD89'],B1T25['ReVC'],B1T25['ReVD'],B1T25['tReVD']
#                          ],axis=1)
# 将B1T25前面的索引换成从0开始，方便画容量
B1T25.reset_index(drop=True, inplace=True)

#首先归一化，再移位处理，得到训练样本
scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_1_T25 = scaler.fit_transform(feature_1_T25)
# # 使用 PCA 进行降维
# pca = PCA(n_components=4)
#scaledFeature_1_T25 = pca.fit_transform(scaledFeature_1_T25)
# 创建包含降维结果的新 DataFrame
scaledFeature_1_T25 = pd.DataFrame(data=scaledFeature_1_T25)
#把T25的特征转化为时间序列
n_steps_in =3 #历史时间长度
n_steps_out=1#预测时间长度
processedFeature_1_T25 = time_series_to_supervised(scaledFeature_1_T25,n_steps_in,n_steps_out)
#print(processedFeature_1_T25.head())
#把T25的容量归一化并转化为时间序列
#首先归一化，再移位处理，得到训练样本
scaler = MinMaxScaler(feature_range=(0, 1))
scaledCapacity_1_T25 = scaler.fit_transform(pd.DataFrame(B1T25['Capacity']))
#print(scaledCapacity_1_T25.shape)
n_steps_in =3 #历史时间长度
n_steps_out=1#预测时间长度
processedCapacity_1_T25 = time_series_to_supervised(scaledCapacity_1_T25,n_steps_in,n_steps_out)
#print(processedCapacity_1_T25.head())
#切片
data_x1 = processedFeature_1_T25.loc[:,'0(t-3)':'7(t-1)']
data_y1=processedCapacity_1_T25.loc[:,'0']
data_y1=data_y1.values.reshape(-1,1)
#划分训练集和测试集
train_X1=data_x1.values[:train_samples]#注意dataframe用reshape,要先用values
test_X1=data_x1.values[train_samples:]
train_y1=data_y1[:train_samples]
test_y1=data_y1[train_samples:]
train_X1 = train_X1.reshape((train_X1.shape[0], n_steps_in, 8))
test_X1 = test_X1.reshape((test_X1.shape[0], n_steps_in, 8))
#print(train_X1.shape, train_y1.shape, test_X1.shape, test_y1.shape)
#%%time
#不进行微调
yhat1 = model.predict(test_X1)
test_y1=test_y1.reshape(-1,1)
inv_forecast_y1 = scaler.inverse_transform(yhat1)
inv_test_y1 = scaler.inverse_transform(test_y1)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

mape_1 = mape(inv_test_y1, inv_forecast_y1)
print('Test MAPE: %.3f' % mape_1)
start_index = 800
mape_1_after_220 = mape(inv_test_y1[start_index:], inv_forecast_y1[start_index:])
print('Test MAPE (starting from cycle 220): %.3f' % mape_1_after_220)
#画图
# # 设置中文显示
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
# # 设置英文显示为 Times New Roman
# matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(8,6))
plt.plot(B1T25['Capacity'], label='True')
x_range = range(train_samples, train_samples + len(inv_forecast_y1))
plt.plot(x_range,inv_forecast_y1,marker='.',label='LSTM',linestyle=None,markersize=5)
plt.xlim(0,1400)
plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
# 在120周期处添加虚线
plt.axvline(x=train_samples, color='gray', linestyle='--')

# 添加箭头
plt.annotate('prediction starting point(cycle=train_samples)',xy=(train_samples, B1T25['Capacity'].iloc[train_samples]), xytext=(train_samples, B1T25['Capacity'].iloc[120]),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, color='black')
# 设置刻度的字体大小和粗细
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 设置图例字体大小和粗细
plt.legend(fontsize=12)
plt.show()
#print('训练完成后权重分配为：%s\n' % (model.layers[2].get_weights()))  # 再次查看训练好的模型中的权重


# In[11]:


#实验24 55到25
# model1=model
# #进行微调
# #冻结前两层
# start_time = time.time()
# for layer in model1.layers[:2]:
#     layer.trainable = False
# #降低学习率
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# model1.compile(loss='mse',optimizer=optimizer)
# #fine-tune
# model1.fit(train_X1, train_y1, epochs=10, batch_size=64, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)

from keras.models import Model
from tensorflow.keras.layers import Input
#这里运行不成功
# for layer in model.layers[:2]:
#         layer.trainable = False
# model.compile(loss='mse', optimizer='adam')
#     # fine tune the model
# transfer_model=model.fit(train_X1, train_y1, epochs=5, batch_size=32,validation_data=(test_X1, test_y1),verbose=2, shuffle=False)

# 冻结两个LSTM层
for layer in model.layers[:2]:  # 包括前两个LSTM层
    layer.trainable = False

#创建新的输出层
input_layer = Input(shape=(train_X1.shape[1], train_X1.shape[2]))
lstm_output_1 = model.layers[0](input_layer)  # 使用原模型的第一个LSTM层
lstm_output_2 = model.layers[1](lstm_output_1)  # 使用原模型的第二个LSTM层
new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)#删掉了激活函数效果更好
#new_dense_1 = model.layers[2](lstm_output_2)
new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)
#new_output_layer = model.layers[3](new_dense_1)

# 创建新的模型
transfer_model = Model(inputs=input_layer, outputs=new_output_layer)

# 编译模型
transfer_model.compile(loss='mse', optimizer='adam')

# 在目标域上进行微调
transfer_model.fit(train_X1, train_y1, epochs=20, batch_size=64, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)

#yhat1= model1.predict(test_X1)
yhat1t= transfer_model.predict(test_X1)
test_y1=test_y1.reshape(-1,1) 
#微调后的 记作1t
inv_forecast_y1t = scaler.inverse_transform(yhat1t)
inv_test_y1t = scaler.inverse_transform(test_y1)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
mape_1t = mape(inv_test_y1t, inv_forecast_y1t)
print('Test MAPE: %.3f' % mape_1t)
#计算后20%的误差
start_index = 220
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
mape_1t_after_220 = mape(inv_test_y1t[start_index:], inv_forecast_y1t[start_index:])
print('Test MAPE (starting from cycle 220): %.3f' % mape_1t_after_220)
#计算结束时间
end_time = time.time()
# 计算模型运行时间
elapsed_time = end_time - start_time
# 打印模型运行时间
print(f"模型运行时间：{elapsed_time} 秒")
# rmse = sqrt(mean_squared_error(inv_test_y10, inv_forecast_y10))#对比test和predict
# print('Test RMSE: %.3f' % rmse)
#画图
plt.figure(figsize=(8,6))
plt.plot(B1T25['Capacity'], label='True')
x_range = range(train_samples, train_samples + len(inv_forecast_y1t))
plt.plot(x_range,inv_forecast_y1t,marker='.',label='LSTM+Fine-tune',linestyle=None,markersize=5)
plt.xlim(0,1400)
plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
# 在120周期处添加虚线
plt.axvline(x=train_samples, color='gray', linestyle='--')

# 添加箭头
import matplotlib
plt.annotate('prediction starting point(cycle=120)',xy=(train_samples, B1T25['Capacity'].iloc[train_samples]), xytext=(train_samples, B1T25['Capacity'].iloc[120]),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, color='black')
# 设置刻度的字体大小和粗细
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 设置图例字体大小和粗细
plt.legend(fontsize=12)
plt.show()
#print('训练完成后权重分配为：%s\n' % (model.layers[2].get_weights()))  # 再次查看训练好的模型中的权重


# In[ ]:


#True-Pre Curve
#LSTM
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def time_series_to_supervised(data, n_in=1, n_out=1,dropnan=True):
    """
    :param data:作为列表或2D NumPy数组的观察序列。需要。
    :param n_in:作为输入的滞后观察数（X）。值可以在[1..len（数据）]之间可选。默认为1。
    :param n_out:作为输出的观测数量（y）。值可以在[0..len（数据）]之间。可选的。默认为1。
    :param dropnan:Boolean是否删除具有NaN值的行。可选的。默认为True。
    :return:
    """
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

from sklearn.decomposition import PCA
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path_5 = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_42_filtered.xlsx'
selected_columns = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[5:1706]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_5 = scaler.fit_transform(Feature_5)

scaledFeature_5 = pd.DataFrame(data=scaledFeature_5)

print(scaledFeature_5.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_5 = time_series_to_supervised(scaledFeature_5,n_steps_in,n_steps_out)
scaler = MinMaxScaler(feature_range=(0, 1))
cap=(pd.read_excel(file_path_5).iloc[5:1706,-1]).values.reshape(-1, 1)
scaledCapacity_5 = scaler.fit_transform(cap)
n_steps_in =3 
n_steps_out=1
processedCapacity_5 = time_series_to_supervised(scaledCapacity_5,n_steps_in,n_steps_out)

data_x5 = processedFeature_5.loc[:,'0(t-3)':'6(t-1)']
data_y5=processedCapacity_5.loc[:,'0']
train_X5=data_x5.values[:1706]
train_y5=data_y5[:1706]
train_X5 = train_X5.reshape((train_X5.shape[0], n_steps_in, 7))


# In[ ]:


#Train source model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from tensorflow.keras.models import save_model, load_model
import os
import time

np.random.seed(42)
tf.random.set_seed(42)

start_time=time.time()

model = Sequential()
model.add(LSTM(96, return_sequences=True, input_shape=(train_X5.shape[1], train_X5.shape[2]))) 

model.add(LSTM(64, return_sequences=False))

model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

history = model.fit(train_X5, train_y5, epochs=30, batch_size=64,verbose=2, shuffle=False)

end_time = time.time()
elapsed_time = end_time - start_time
# 打印模型运行时间
#print(f"模型运行时间：{elapsed_time} 秒")


# In[4]:


# T45 Validation no finetune
import matplotlib
train_samples=60

selected_columns = [ 'Vg1','Vg2','Q2','RL1', 'RL2', 'RO', 'tVD2']
file_path_6 = r'D:\桌面上的文件\Data Sufficiency\HUST\Features_HUST_82_filtered.xlsx'  
Feature_6 = pd.read_excel(file_path_6, usecols=selected_columns).iloc[5:2283]
capacity_6=pd.read_excel(file_path_6).iloc[5:2283,-1]
cap_6=capacity_6.values.reshape(-1, 1)


#Feature_6 = pd.read_excel(file_path_6, usecols=selected_columns).iloc[:800]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_6 = scaler.fit_transform(Feature_6)
scaledFeature_6 = pd.DataFrame(data=scaledFeature_6)

n_steps_in =3 
n_steps_out=1
processedFeature_6 = time_series_to_supervised(scaledFeature_6,n_steps_in,n_steps_out)
scaler = MinMaxScaler(feature_range=(0, 1))
#cap_6=(pd.read_excel(file_path_6).iloc[:800,-1]).values.reshape(-1, 1)
scaledCapacity_6 = scaler.fit_transform(cap_6)

n_steps_in =3 
n_steps_out=1
processedCapacity_6 = time_series_to_supervised(scaledCapacity_6,n_steps_in,n_steps_out)
data_x6 = processedFeature_6.loc[:,'0(t-3)':'6(t-1)']
data_y6=processedCapacity_6.loc[:,'0']
data_y6=data_y6.values.reshape(-1,1)
train_X6=data_x6.values[:train_samples]
test_X6=data_x6.values[train_samples:]
train_y6=data_y6[:train_samples]
test_y6=data_y6[train_samples:]
train_X6 = train_X6.reshape((train_X6.shape[0], n_steps_in, 7))
test_X6 = test_X6.reshape((test_X6.shape[0], n_steps_in, 7))

yhat6 = model.predict(test_X6)
test_y6=test_y6.reshape(-1,1)
inv_forecast_y6 = scaler.inverse_transform(yhat6)
inv_test_y6 = scaler.inverse_transform(test_y6)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

mape_6 = mape(inv_test_y6, inv_forecast_y6)
print('Test MAPE: %.3f' % mape_6)

plt.figure(figsize=(8,6))
plt.plot(pd.read_excel(file_path_6).iloc[:2283,-1], label='True')
x_range = range(train_samples, train_samples+ len(inv_forecast_y6))
plt.plot(x_range,inv_forecast_y6,marker='.',label='LSTM',linestyle=None,markersize=5)

plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[5]:


# T45 Validation with finetune
start_time = time.time()

from keras.models import Model
from tensorflow.keras.layers import Input

for layer in model.layers[:2]:  
    layer.trainable = False

input_layer = Input(shape=(train_X6.shape[1], train_X6.shape[2]))
lstm_output_1 = model.layers[0](input_layer)  
lstm_output_2 = model.layers[1](lstm_output_1)  
new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)
new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)

transfer_model = Model(inputs=input_layer, outputs=new_output_layer)
transfer_model.compile(loss='mse', optimizer='adam')
transfer_model.fit(train_X6, train_y6, epochs=50, batch_size=64, verbose=2, shuffle=False)

yhat6t= transfer_model.predict(test_X6)
test_y6=test_y6.reshape(-1,1) 
inv_forecast_y6t = scaler.inverse_transform(yhat6t)
inv_test_y6t = scaler.inverse_transform(test_y6)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))
mape_6t = mape(inv_test_y6t, inv_forecast_y6t)
print('Test MAPE: %.3f' % mape_6t)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"模型运行时间：{elapsed_time} 秒")

plt.figure(figsize=(8,6))
plt.plot(pd.read_excel(file_path_6).iloc[:2283,-1], label='True')
x_range = range(train_samples, train_samples+len(inv_forecast_y6t))
plt.plot(x_range,inv_forecast_y6t,marker='.',label='LSTM+Fine-tune',linestyle=None,markersize=5)
plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel(file_path_6)
initial_capacity = data.iloc[0, -1]  
threshold_capacity = 0.95 * initial_capacity 

plt.figure(figsize=(4, 3),dpi=600)
plt.plot(data.iloc[:2283, -1], label='Target_True',linewidth=5,color=plt.cm.Blues(0.8))
plt.plot(pd.read_excel(file_path_5).iloc[5:1706,-1], label='Source',linewidth=5,color=plt.cm.Reds(0.8))


x_range = range(train_samples, train_samples + len(inv_forecast_y6t))
plt.plot(x_range, inv_forecast_y6t,label='Target_Pre', linestyle=None, linewidth=5,color=plt.cm.Greens(0.8))

def find_intersection(x_vals, y_vals, threshold):
    for i in range(len(y_vals) - 1):
        if (y_vals[i] >= threshold and y_vals[i + 1] < threshold) or (y_vals[i] <= threshold and y_vals[i + 1] > threshold):
            return x_vals[i]  
    return None

true_x_intersection = train_samples+find_intersection(range(len(inv_test_y6t)), inv_test_y6t, threshold_capacity)
pred_x_intersection = find_intersection(x_range, inv_forecast_y6t.flatten(), threshold_capacity)

plt.ylabel('Capacity(mAh)', fontsize=12)
plt.xlabel('Cycle', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('HUST-3C')
plt.show()

print(f"真实容量曲线与 0.8 初始容量的交点对应的循环数: {true_x_intersection}")
print(f"预测容量曲线与 0.8 初始容量的交点对应的循环数: {pred_x_intersection}")

