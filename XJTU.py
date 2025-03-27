#!/usr/bin/env python
# coding: utf-8

# In[31]:


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

file_path = r'D:\桌面上的文件\Data Sufficiency\Xjtu\Batch-1\2C_battery-1.mat'  
mat_data = scipy.io.loadmat(file_path)

struct_data = mat_data['data']  
num_cycles = struct_data.shape[1]  


cmap = cm.get_cmap('Blues', num_cycles)  
colors = cmap(np.linspace(0, 1, num_cycles))


plt.figure(figsize=(4, 3), dpi=600)


for idx in range(num_cycles):
    cycle_data = struct_data[0, idx]  
    
    voltage = cycle_data['voltage_V'].flatten()  
    capacity = cycle_data['capacity_Ah'].flatten()  
    
    closest_idx = np.argmin(np.abs(voltage - 4.185))
     
    voltage = voltage[:closest_idx + 1]
    capacity = capacity[:closest_idx + 1]
   
    plt.plot(capacity, voltage, linewidth=1, color=colors[idx], label=f'Cycle {idx + 1}' if idx % 20 == 0 else "")

sm = cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=1, vmax=num_cycles))


plt.title('Charging Voltage vs Capacity for Each Cycle')
plt.xlabel('Charge Capacity (Ah)')
plt.ylabel('Voltage (V)')
plt.ylim(2, 4.25)
plt.tight_layout()
plt.show()


# In[32]:


#XJTU Batch1 Batch2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


file1 = r'D:\桌面上的文件\Data Sufficiency\Xjtu\Battery_Features_b1.xlsx'
file2 = r'D:\桌面上的文件\Data Sufficiency\Xjtu\Battery_Features_b2.xlsx'

sheets_file1 = pd.ExcelFile(file1).sheet_names
sheets_file2 = pd.ExcelFile(file2).sheet_names

plt.figure(figsize=(4, 3), dpi=600)

for i, sheet in enumerate(sheets_file1):
    data = pd.read_excel(file1, sheet_name=sheet, header=None)
    column_data = data.iloc[5:, 5]  
    plt.plot(column_data, label=f'File1 - {sheet}', marker='o',markersize=3,color=cm.Blues(i / len(sheets_file1)))

for i, sheet in enumerate(sheets_file2):
    data = pd.read_excel(file2, sheet_name=sheet, header=None)
    column_data = data.iloc[5:, 5]  
    plt.plot(column_data, label=f'File2 - {sheet}',marker='o',markersize=3, color=cm.Reds(i / len(sheets_file2)))

plt.title('XJTU', fontsize=12)
plt.xlabel('Index', fontsize=10)
plt.ylabel('Value (Fifth Column)', fontsize=10)
#plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
#plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


# In[ ]:


# XJTU-PC-Batch1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'E:\Batch-1\Battery_Features_b1.xlsx'  

battery_sheet_Batch1 = ['Battery_1', 'Battery_2', 'Battery_3', 'Battery_4', 
                        'Battery_5', 'Battery_6', 'Battery_7', 'Battery_8']

periods = range(5, 201, 5)
all_correlations = []

for sheet_name in battery_sheet_Batch1:
    
    Feature_B1 = pd.read_excel(file_path, sheet_name=sheet_name).iloc[3:, 0:5]  
    dis_cap = pd.read_excel(file_path, sheet_name=sheet_name).iloc[3:, 5]  
    
    correlation_trend1 = []
 
    for col, feature_data in Feature_B1.items():
        correlations1 = []  

        for seg in periods:
          
            feature_slice = feature_data.iloc[:seg]
            capacity_slice = dis_cap.iloc[:seg]  

            correlation = feature_slice.corr(capacity_slice)

            correlations1.append(abs(correlation))

        correlation_trend1.append(correlations1)
    
    all_correlations.append(np.array(correlation_trend1))

all_correlations = np.array(all_correlations)

average_correlations = np.mean(all_correlations, axis=0)

average_correlations_overall = np.mean(average_correlations, axis=0)

plt.figure(figsize=(5, 3))
for i in range(average_correlations.shape[0]):
    plt.plot(periods, average_correlations[i], label=f'Feature {i+1}')

plt.plot(periods, average_correlations_overall, 'o-', color='black', label='Average')
plt.xlabel('Cycles')
plt.ylabel('PC (Correlation)')
plt.title('Average Correlation Trend of Features with Capacity for Batch1')
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


# XJTU-PC-Batch2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'E:\Batch-2\Battery_Features_b2.xlsx'  

battery_sheet_Batch2 = ['Battery_1', 'Battery_2', 'Battery_3', 'Battery_4', 
                        'Battery_5', 'Battery_6', 'Battery_7', 'Battery_8',
                        'Battery_9', 'Battery_10', 'Battery_11', 'Battery_12',
                        'Battery_13', 'Battery_14', 'Battery_15']

periods = range(5, 201, 5)
all_correlations = []

for sheet_name in battery_sheet_Batch2:
  
    Feature_B1 = pd.read_excel(file_path, sheet_name=sheet_name).iloc[3:, 0:5] 
    dis_cap = pd.read_excel(file_path, sheet_name=sheet_name).iloc[3:, 5]  
    
    correlation_trend1 = []
  
    for col, feature_data in Feature_B1.items():
        correlations1 = []  

        for seg in periods:
            
            feature_slice = feature_data.iloc[:seg]
            capacity_slice = dis_cap.iloc[:seg]  

            correlation = feature_slice.corr(capacity_slice)

            correlations1.append(abs(correlation))

        correlation_trend1.append(correlations1)
 
    all_correlations.append(np.array(correlation_trend1))

all_correlations = np.array(all_correlations)

average_correlations2 = np.mean(all_correlations, axis=0)

average_correlations_overall2 = np.mean(average_correlations2, axis=0)

plt.figure(figsize=(5, 3))
for i in range(average_correlations.shape[0]):
    plt.plot(periods, average_correlations2[i], label=f'Feature {i+1}')

plt.plot(periods, average_correlations_overall2, 'o-', color='black', label='Average')
plt.xlabel('Cycles')
plt.ylabel('PC (Correlation)')
plt.title('Average Correlation Trend of Features with Capacity for Batch2')
plt.legend()
plt.grid(True)
plt.show()


# In[34]:


# XJTU TC 2 to 1 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

file_path_Batch1 = r'D:\桌面上的文件\Data Sufficiency\Xjtu\Battery_Features_b1.xlsx'
file_path_Batch2 = r'D:\桌面上的文件\Data Sufficiency\Xjtu\Battery_Features_b2.xlsx'

battery_sheet_Batch1 = ['Battery_1', 'Battery_2', 'Battery_3', 'Battery_4', 'Battery_5', 'Battery_6', 'Battery_7', 'Battery_8']
battery_sheet_Batch2 = ['Battery_1', 'Battery_2', 'Battery_3', 'Battery_4', 'Battery_5', 'Battery_6', 'Battery_7', 'Battery_8', 
                        'Battery_9', 'Battery_10', 'Battery_11', 'Battery_12', 'Battery_13', 'Battery_14', 'Battery_15']

feature_names = ['Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']

periods = list(range(5, 201, 5))

wasserstein_distances_per_feature = {feature: [] for feature in feature_names}

for sheet_Batch1 in battery_sheet_Batch1:
    for sheet_Batch2 in battery_sheet_Batch2:

        df_battery_Batch1 = pd.read_excel(file_path_Batch1, sheet_name=sheet_Batch1).iloc[2:, 0:5]  
        df_battery_Batch2 = pd.read_excel(file_path_Batch2, sheet_name=sheet_Batch2).iloc[2:, 0:5]  

        feature_Batch1 = df_battery_Batch1[feature_names]
        feature_Batch2 = df_battery_Batch2[feature_names]

        scaler = MinMaxScaler(feature_range=(0, 1))
        feature_Batch1_normalized = scaler.fit_transform(feature_Batch1)
        feature_Batch2_normalized = scaler.fit_transform(feature_Batch2)

        for feature_idx, feature in enumerate(feature_names):    
            distances = []
            
            for period in periods:
              
                subset_Batch1 = feature_Batch1_normalized[:period, feature_idx]
                subset_Batch2 = feature_Batch2_normalized[:period, feature_idx]

                w_distance = wasserstein_distance(subset_Batch1.flatten(), subset_Batch2.flatten())
                distances.append(1 - w_distance)  

            wasserstein_distances_per_feature[feature].append(distances)

average_wasserstein_distances_per_feature = {}
for feature, distances_list in wasserstein_distances_per_feature.items():
    distances_array = np.array(distances_list)
    average_distances = np.mean(distances_array, axis=0)
    average_wasserstein_distances_per_feature[feature] = average_distances

plt.figure(figsize=(5, 3), dpi=300)
for feature, avg_distances in average_wasserstein_distances_per_feature.items():
    plt.plot(periods, avg_distances, marker='o', label=feature)

plt.title('Wasserstein Distance Between Batch1 and Batch2 for Each Feature')
plt.xlabel('Cycles')
plt.ylabel('Average Transferable Capability (1 - W Distance)')
plt.grid(True)
plt.legend(title='Features')
plt.show()


# In[11]:


import pandas as pd
import numpy as np

df_wasserstein = pd.DataFrame(average_wasserstein_distances_per_feature)
df_correlation = pd.DataFrame(average_correlations.T, columns=[f'Feature_{i+1}' for i in range(5)])
df_correlation2 = pd.DataFrame(average_correlations2.T, columns=[f'Feature_{i+1}' for i in range(5)])

file_path = 'D:\\桌面上的文件\\XJTU.xlsx'
with pd.ExcelWriter(file_path) as writer:
    df_wasserstein.to_excel(writer, sheet_name='TC_Trend', index=False)
    df_correlation.to_excel(writer, sheet_name='PC_Trend', index=False)
    df_correlation2.to_excel(writer, sheet_name='PC_Trend_Batch2', index=False)

print(f"数据已成功写入 {file_path}")


# In[9]:


#DS Validation
#LSTM Prediction
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from numpy import concatenate
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

file_path = r'D:\桌面上的文件\Data Sufficiency\Xjtu\Battery_Features_b2.xlsx'  
selected_columns = [ 'Vg', 'Q', 'RL', 'T_CCCV','T_CVCA']
Feature_B2 = pd.read_excel(file_path,sheet_name='Battery_11',usecols=selected_columns).iloc[3:135,]
cap_B2= pd.read_excel(file_path,sheet_name='Battery_11').iloc[3:135,5]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_B2 = scaler.fit_transform(Feature_B2)

scaledFeature_B2 = pd.DataFrame(data=scaledFeature_B2)

print(scaledFeature_B2.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_B2 = time_series_to_supervised(scaledFeature_B2,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
cap_B2 = cap_B2.values.reshape(-1, 1) 
scaledCapacity_B2 = scaler.fit_transform(cap_B2)

n_steps_in =3 
n_steps_out=1
processedCapacity_B2 = time_series_to_supervised(scaledCapacity_B2,n_steps_in,n_steps_out)

data_xB2 = processedFeature_B2.loc[:,'0(t-3)':'4(t-1)']
data_yB2=processedCapacity_B2.loc[:,'0']
data_yB2=data_yB2.values.reshape(-1,1)
train_XB2=data_xB2.values[:300]
train_yB2=data_yB2[:300]
train_XB2 = train_XB2.reshape((train_XB2.shape[0], n_steps_in, 5))


# In[10]:


#Train Source Model
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
model.add(LSTM(96, return_sequences=True, input_shape=(train_XB2.shape[1], train_XB2.shape[2]))) 
model.add(LSTM(64, return_sequences=False))

model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

history = model.fit(train_XB2, train_yB2, epochs=30, batch_size=64,verbose=2, shuffle=False)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"模型运行时间：{elapsed_time} 秒")


# In[4]:


# Batch2 transfer to Batch1，look at prediction accuracy in different fine-tuning cycles automaticly 
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

selected_columns = [ 'Vg', 'Q', 'RL', 'T_CCCV','T_CVCA']
file_path_B2 = r'E:\Batch-2\Battery_Features_b2.xlsx'  
Feature_B2 = pd.read_excel(file_path_B2,sheet_name='Battery_1').iloc[3:300,:5]
cap_B2= pd.read_excel(file_path_B2,sheet_name='Battery_1').iloc[3:300,5]


file_path_B1 = r'E:\Batch-1\Battery_Features_b1.xlsx'  
Feature_B1 = pd.read_excel(file_path_B1,sheet_name='Battery_1',usecols=selected_columns).iloc[3:420,]
cap_B1= pd.read_excel(file_path_B1,sheet_name='Battery_1').iloc[3:420,5]
cap_B1 = cap_B1.values.reshape(-1, 1) 
n_steps_in=3
n_steps_out = 1
np.random.seed(42)
tf.random.set_seed(42)
prediction_capability=[]
for train_samples in range(5,101,5):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaledFeature_B1 = scaler.fit_transform(Feature_B1)
    scaledFeature_B1 = pd.DataFrame(data=scaledFeature_B1)
    n_steps_in =3 
    n_steps_out=1
    processedFeature_B1 = time_series_to_supervised(scaledFeature_B1,n_steps_in,n_steps_out)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledCapacity_B1 = scaler.fit_transform(cap_B1)

    n_steps_in =3 
    n_steps_out=1
    processedCapacity_B1 = time_series_to_supervised(scaledCapacity_B1,n_steps_in,n_steps_out)
    data_xB1 = processedFeature_B1.loc[:,'0(t-3)':'4(t-1)']
    data_yB1=processedCapacity_B1.loc[:,'0']
    data_yB1=data_yB1.values.reshape(-1,1)
    train_XB1=data_xB1.values[:train_samples]
    test_XB1=data_xB1.values[train_samples:]
    train_yB1=data_yB1[:train_samples]
    test_yB1=data_yB1[train_samples:]
    train_XB1 = train_XB1.reshape((train_XB1.shape[0], n_steps_in, 5))
    test_XB1 = test_XB1.reshape((test_XB1.shape[0], n_steps_in, 5))
    for layer in model.layers[:2]:  
        layer.trainable = False

    input_layer = Input(shape=(train_XB1.shape[1], train_XB1.shape[2]))
    lstm_output_1 = model.layers[0](input_layer) 
    lstm_output_2 = model.layers[1](lstm_output_1)  
    new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)
    new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)
    transfer_model = Model(inputs=input_layer, outputs=new_output_layer)
    transfer_model.compile(loss='mse', optimizer='adam')
    transfer_model.fit(train_XB1, train_yB1, epochs=50, batch_size=64, verbose=2, shuffle=False)   
    yhatB1t= transfer_model.predict(test_XB1)
    test_yB1=test_yB1.reshape(-1,1) 
    inv_forecast_yB1t = scaler.inverse_transform(yhatB1t)
    inv_test_yB1t = scaler.inverse_transform(test_yB1)
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true))
    mape_B1t = mape(inv_test_yB1t, inv_forecast_yB1t)
    print('Test MAPE: %.3f' % mape_B1t)
    prediction_capability.append(1-mape_B1t)
prediction_capability


# In[6]:


# XJTU Acc
import pandas as pd

prediction_df = pd.DataFrame(prediction_capability, columns=["Batch2 to Batch1"])

file_path_output = "D:\桌面上的文件\XJTU_Acc.xlsx"

prediction_df.to_excel(file_path_output, index=False)


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt

file_path = 'D:/桌面上的文件/XJTU_Acc.xlsx'  
df = pd.read_excel(file_path)

column_data = df['Batch2 to Batch1']

column_data = df['Batch2 to Batch1']

plt.figure(figsize=(5, 2), dpi=600)

for i in range(len(column_data)):
    if i == 1:  # 第四个柱子
        plt.bar(i, column_data[i], width=0.8, color='#898988')
    else:
        plt.bar(i, column_data[i], width=0.8, color='#d7d7d7')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.grid(False)
plt.ylim(0.98,0.998)
plt.show() 


# In[12]:


#PC TC Trend
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\XJTU.xlsx'  
df_TC = pd.read_excel(file_path, sheet_name='TC_Trend') 
df_PC = pd.read_excel(file_path, sheet_name='PC_Trend')  

mean_wasserstein = df_TC.mean(axis=1)
mean_correlation = df_PC.mean(axis=1)

fig, ax1 = plt.subplots(figsize=(4, 2), dpi=600)
color = '#599CB4'  
ax1.set_xlabel('Periods')
ax1.set_ylabel('Average Wasserstein Trend', color=color)
ax1.plot(mean_wasserstein[:20], marker='o', markersize=9,linestyle='-', linewidth=4,color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.8, 1) 
ax2 = ax1.twinx()  
color = '#C25759' 
ax2.set_ylabel('Average Correlation Trend', color=color)
ax2.plot(mean_correlation[:20], marker='o', markersize=9,linestyle='-', linewidth=4,color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.4, 1)  
plt.grid(False)
plt.show()


# In[ ]:


#True-Pre Curve
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from numpy import concatenate
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

file_path = r'D:\桌面上的文件\Data Sufficiency\Xjtu\Battery_Features_b2.xlsx'  
selected_columns = [ 'Vg', 'Q', 'RL', 'T_CCCV','T_CVCA']
Feature_B2 = pd.read_excel(file_path,sheet_name='Battery_11',usecols=selected_columns).iloc[3:135,]
cap_B2= pd.read_excel(file_path,sheet_name='Battery_11').iloc[3:135,5]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_B2 = scaler.fit_transform(Feature_B2)

scaledFeature_B2 = pd.DataFrame(data=scaledFeature_B2)

print(scaledFeature_B2.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_B2 = time_series_to_supervised(scaledFeature_B2,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
cap_B2 = cap_B2.values.reshape(-1, 1) 
scaledCapacity_B2 = scaler.fit_transform(cap_B2)

n_steps_in =3 
n_steps_out=1
processedCapacity_B2 = time_series_to_supervised(scaledCapacity_B2,n_steps_in,n_steps_out)

data_xB2 = processedFeature_B2.loc[:,'0(t-3)':'4(t-1)']
data_yB2=processedCapacity_B2.loc[:,'0']
data_yB2=data_yB2.values.reshape(-1,1)
train_XB2=data_xB2.values[:300]
train_yB2=data_yB2[:300]
train_XB2 = train_XB2.reshape((train_XB2.shape[0], n_steps_in, 5))


# In[ ]:


#Train Source Model
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
model.add(LSTM(96, return_sequences=True, input_shape=(train_XB2.shape[1], train_XB2.shape[2]))) 
model.add(LSTM(64, return_sequences=False))

model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

history = model.fit(train_XB2, train_yB2, epochs=30, batch_size=64,verbose=2, shuffle=False)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"模型运行时间：{elapsed_time} 秒")


# In[11]:


# Batch1 Validation no finetune
import matplotlib
train_samples=5

selected_columns = ['Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']
file_path_6 = r'D:\桌面上的文件\Data Sufficiency\Xjtu\Battery_Features_b1.xlsx'  
Feature_6 = pd.read_excel(file_path_6,sheet_name='Battery_8',usecols=selected_columns).iloc[3:420,0:5]
capacity_6=pd.read_excel(file_path_6,sheet_name='Battery_8').iloc[3:420,5]
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
data_x6 = processedFeature_6.loc[:,'0(t-3)':'4(t-1)']
data_y6=processedCapacity_6.loc[:,'0']
data_y6=data_y6.values.reshape(-1,1)
train_X6=data_x6.values[:train_samples]
test_X6=data_x6.values[train_samples:]
train_y6=data_y6[:train_samples]
test_y6=data_y6[train_samples:]
train_X6 = train_X6.reshape((train_X6.shape[0], n_steps_in, 5))
test_X6 = test_X6.reshape((test_X6.shape[0], n_steps_in, 5))

yhat6 = model.predict(test_X6)
test_y6=test_y6.reshape(-1,1)
inv_forecast_y6 = scaler.inverse_transform(yhat6)
inv_test_y6 = scaler.inverse_transform(test_y6)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

mape_6 = mape(inv_test_y6, inv_forecast_y6)
print('Test MAPE: %.3f' % mape_6)

plt.figure(figsize=(8,6))
plt.plot(pd.read_excel(file_path_6, sheet_name='Battery_8').iloc[3:420,-1], label='True')
x_range = range(train_samples, train_samples+ len(inv_forecast_y6))
plt.plot(x_range,inv_forecast_y6,marker='.',label='LSTM',linestyle=None,markersize=5)

plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[12]:


# B1 Validation with finetune
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
plt.plot(pd.read_excel(file_path_6,sheet_name='Battery_8').iloc[3:420,-1], label='True')
x_range = range(train_samples, train_samples+len(inv_forecast_y6t))
plt.plot(x_range,inv_forecast_y6t,marker='.',label='LSTM+Fine-tune',linestyle=None,markersize=5)
plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[42]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel(file_path_6,sheet_name='Battery_8')
initial_capacity = data.iloc[0, -1]  
threshold_capacity = 0.95 * initial_capacity  

plt.figure(figsize=(4, 3),dpi=600)
plt.plot(data.iloc[3:420, -1], label='Target_True',linewidth=5,color=plt.cm.Blues(0.8))
plt.plot(cap_B2,label='Source',linewidth=5,color=plt.cm.Reds(0.8))

x_range = range(train_samples, train_samples + len(inv_forecast_y6t))
plt.plot(inv_forecast_y6t, label='Target_Pre', linestyle=None, linewidth=5,color=plt.cm.Greens(0.8))

def find_intersection(x_vals, y_vals, threshold):
    for i in range(len(y_vals) - 1):
        if (y_vals[i] >= threshold and y_vals[i + 1] < threshold) or (y_vals[i] <= threshold and y_vals[i + 1] > threshold):
            return x_vals[i]  
    return None

true_x_intersection = train_samples+find_intersection(range(len(inv_test_y6t)), inv_test_y6t, threshold_capacity)
pred_x_intersection = find_intersection(x_range, inv_forecast_y6t.flatten(), threshold_capacity)

plt.ylabel('Capacity(Ah)', fontsize=12)
plt.xlabel('Cycle', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('XJTU-2C')
plt.show()

print(f"真实容量曲线与 0.8 初始容量的交点对应的循环数: {true_x_intersection}")
print(f"预测容量曲线与 0.8 初始容量的交点对应的循环数: {pred_x_intersection}")

