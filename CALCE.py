#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Voltage-SOC
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

folder_path = r'C:\Users\86180\Downloads\CS2_35'

files = [(f, os.path.getmtime(os.path.join(folder_path, f))) for f in os.listdir(folder_path) if f.endswith('.xlsx')]

files.sort(key=lambda x: x[1])

sheet_names = ['Channel_1-006', 'Channel_1-008']

cumulative_Q = 0

plt.figure(figsize=(4, 3), dpi=600)

colors = cm.Blues(np.linspace(0.2, 1, len(files)))
#colors = cm.Blues(np.linspace(0.2, 1, 2000))

for file_idx, (file, _) in enumerate(files):
    file_path = os.path.join(folder_path, file)
    data = None

    for sheet_name in sheet_names:
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            break  
        except:
            continue  
   
    if data is None:
        print(f"文件 {file} 中没有找到指定的Sheet")
        continue

    data = data[data['Cycle_Index'] >= 2]

    for cycle_num, cycle_data in data.groupby('Cycle_Index'):
        filtered_data = cycle_data[cycle_data['Step_Index'] == 2]

        if not filtered_data.empty:
            charging_voltage = filtered_data['Voltage(V)'].values
            charging_voltage_time = filtered_data['Test_Time(s)'].values
            charging_capacity = filtered_data['Charge_Capacity(Ah)'].values
            charging_capacity = charging_capacity - charging_capacity[0]
  
            plt.plot(charging_capacity, charging_voltage, 
                     linewidth=1, color=colors[file_idx])

plt.title('Charging Voltage-SOC Each Cycle', fontsize=10)
plt.xlabel('Cumulative Charge Capacity (Ah)', fontsize=10)
plt.ylabel('Voltage (V)', fontsize=10)
plt.ylim(2, 4.25)
plt.tight_layout()
#plt.grid(alpha=0.3)
plt.show()


# In[6]:


#CALCE capacity
import pandas as pd
import matplotlib.pyplot as plt
import os

file1 = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-33-0_5.xlsx'
file2 = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-35-1.xlsx'

data1 = pd.read_excel(file1, header=None)  
capacity1 = data1.iloc[1:, 6]  

data2 = pd.read_excel(file2, header=None) 
capacity2 = data2.iloc[1:, 6] 

plt.figure(figsize=(4, 3), dpi=600)
plt.plot(capacity2, label='File 2', marker='o', markersize=3, color=plt.cm.Reds(0.8))
plt.plot(capacity1, label='File 1', marker='o', markersize=3, color=plt.cm.Blues(0.8))

plt.title('Capacity in CALCE', fontsize=10)
plt.xlabel('Cycle Index', fontsize=10)
plt.ylabel('Capacity (Ah)', fontsize=10)
plt.tight_layout()
plt.show()


# In[1]:


#CS_35 Feature extraction Different excels cumulative
import pandas as pd
import numpy as np
import os
from datetime import datetime

folder_path = r'C:\Users\86180\Downloads\CS2_35'

files = [(f, os.path.getmtime(os.path.join(folder_path, f))) for f in os.listdir(folder_path) if f.endswith('.xlsx')]

files.sort(key=lambda x: x[1])

results = []

sheet_names = ['Channel_1-006', 'Channel_1-008']

cumulative_Q = 0

for file, _ in files:
    file_path = os.path.join(folder_path, file)
    data = None

    for sheet_name in sheet_names:
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            break  
        except:
            continue  

    if data is None:
        print(f"文件 {file} 中没有找到指定的Sheet")
        continue

    file_cumulative_Q = cumulative_Q

    for cycle_num, cycle_data in data.groupby('Cycle_Index'):
     
        filtered_data = cycle_data[(cycle_data['Step_Index'] == 2)]
        filtered_data2 = cycle_data[(cycle_data['Step_Index'] == 4)]
        filtered_data3 = cycle_data[(cycle_data['Step_Index'] == 7)]

        if not filtered_data.empty and not filtered_data2.empty and not filtered_data3.empty:
          
            charging_voltage = filtered_data['Voltage(V)'].values
            charging_voltage_time = filtered_data['Test_Time(s)'].values
            charging_capacity = filtered_data2['Charge_Capacity(Ah)'].values
            charging_current = filtered_data2['Current(A)'].values
            charging_current_time = filtered_data2['Test_Time(s)'].values
            discharging_current = filtered_data3['Current(A)'].values
            discharging_current_time = filtered_data3['Test_Time(s)'].values

            index = np.argmin(np.abs(charging_voltage - 4.0))
            index_current = np.argmin(np.abs(charging_current - 1.0))
           
            closest_voltage = charging_voltage[index]
            closest_current = charging_current[index_current]
            
            Gra = np.diff(charging_voltage) / np.diff(charging_voltage_time)
            Vg = np.mean(Gra)
            Q = charging_capacity[-1]
            RL = (charging_voltage[-1] - charging_voltage[0]) / 0.55
            T_CCCV = charging_voltage_time[-1] - charging_voltage_time[index]  # 电压从4.0到4.2所需要的时间
            T_CVCA = charging_current_time[-1] - charging_current_time[index_current]
            Capacity = -np.trapz(discharging_current, discharging_current_time)

            results.append([cycle_num, Vg, file_cumulative_Q + Q, RL, T_CCCV, T_CVCA, Capacity])

    cumulative_Q = file_cumulative_Q + Q

result_df = pd.DataFrame(results, columns=['cycle number', 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA', 'Capacity'])
print(result_df)

result_df.to_excel(r'D:\桌面上的文件\Features_CS2-35-1.xlsx', index=False)

print("特征提取完成，并已保存")


# In[10]:


#Filtering
import pandas as pd
import numpy as np
from scipy.signal import medfilt

file_path = 'D:\桌面上的文件\Features_CS2-35-1.xlsx'
df = pd.read_excel(file_path)

def apply_median_filter(series, kernel_size=3):
    return medfilt(series, kernel_size=kernel_size)

filtered_df = df.copy()
for column in df.columns:
    filtered_df[column] = apply_median_filter(df[column].values, kernel_size=3)

filtered_file_path = 'D:\桌面上的文件\Features_lb_CS2-35-1.xlsx'
filtered_df.to_excel(filtered_file_path, index=False)


# In[11]:


#CS_33 Feature extraction Different excels cumulative
import pandas as pd
import numpy as np
import os
from datetime import datetime

folder_path = r'E\CS2_35'

files = [(f, os.path.getmtime(os.path.join(folder_path, f))) for f in os.listdir(folder_path) if f.endswith('.xlsx')]

files.sort(key=lambda x: x[1])

results = []

sheet_names = ['Channel_1-006', 'Channel_1-008']

cumulative_Q = 0

for file, _ in files:
    file_path = os.path.join(folder_path, file)
    data = None

    for sheet_name in sheet_names:
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            break  
        except:
            continue  

    if data is None:
        print(f"文件 {file} 中没有找到指定的Sheet")
        continue

    file_cumulative_Q = cumulative_Q

    for cycle_num, cycle_data in data.groupby('Cycle_Index'):
      
        filtered_data = cycle_data[(cycle_data['Step_Index'] == 2)]
        filtered_data2 = cycle_data[(cycle_data['Step_Index'] == 4)]
        filtered_data3 = cycle_data[(cycle_data['Step_Index'] == 7)]

        if not filtered_data.empty and not filtered_data2.empty and not filtered_data3.empty:
            
            charging_voltage = filtered_data['Voltage(V)'].values
            charging_voltage_time = filtered_data['Test_Time(s)'].values
            charging_capacity = filtered_data2['Charge_Capacity(Ah)'].values
            charging_current = filtered_data2['Current(A)'].values
            charging_current_time = filtered_data2['Test_Time(s)'].values
            discharging_current = filtered_data3['Current(A)'].values
            discharging_current_time = filtered_data3['Test_Time(s)'].values

            index = np.argmin(np.abs(charging_voltage - 4.0))
            index_current = np.argmin(np.abs(charging_current - 1.0))
            
            closest_voltage = charging_voltage[index]
            closest_current = charging_current[index_current]
         
            Gra = np.diff(charging_voltage) / np.diff(charging_voltage_time)
            Vg = np.mean(Gra)
            Q = charging_capacity[-1]
            RL = (charging_voltage[-1] - charging_voltage[0]) / 0.55
            T_CCCV = charging_voltage_time[-1] - charging_voltage_time[index]  
            T_CVCA = charging_current_time[-1] - charging_current_time[index_current]
            Capacity = -np.trapz(discharging_current, discharging_current_time)

            results.append([cycle_num, Vg, file_cumulative_Q + Q, RL, T_CCCV, T_CVCA, Capacity])

    cumulative_Q = file_cumulative_Q + Q

result_df = pd.DataFrame(results, columns=['cycle number', 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA', 'Capacity'])
print(result_df)

result_df.to_excel('D:\桌面上的文件\Features_CS2-33-0.5.xlsx', index=False)

print("特征提取完成，并已保存")


# In[12]:


# Filtering
import pandas as pd
import numpy as np
from scipy.signal import medfilt

file_path = 'Features_CS2-33-0_5'
df = pd.read_excel(file_path)

def apply_median_filter(series, kernel_size=3):
    return medfilt(series, kernel_size=kernel_size)

filtered_df = df.copy()
for column in df.columns:
    filtered_df[column] = apply_median_filter(df[column].values, kernel_size=3)

filtered_file_path = 'Features_lb_CS2-33-0_5'
filtered_df.to_excel(filtered_file_path, index=False)


# In[2]:


# PC 33
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-33-0_5.xlsx'
df = pd.read_excel(file_path).iloc[4:,:]

columns_to_analyze = [ 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']

def calculate_correlations(df, cycles, target='Capacity'):
    cycle_data = df[df['cycle number'] <= cycles]
    correlations = cycle_data[columns_to_analyze + [target]].corr()[target].drop(target)
    return abs(correlations)

correlations = {}

for cycle in range(10, len(df)+1,10):
    correlations[cycle] = calculate_correlations(df, cycle)
    
average_correlations = np.mean([list(correlations[cycle].values) for cycle in range(10, len(df)+1,10)], axis=1)

plt.figure(figsize=(10, 6),dpi=600)

for feature in columns_to_analyze:
    corr_values = [correlations[cycle][feature] for cycle in range(10, len(df)+1,10)]
    plt.plot(range(10, len(df)+1,10), corr_values, marker='o', linestyle='-', label=feature)

plt.plot(range(10, len(df)+1,10), average_correlations, marker='o', linestyle='-', color='black', label='Average')

plt.title('Correlations with Capacity (Different Features) CS33')
plt.xlabel('Cycles')
plt.ylabel('Correlation')
plt.xticks(range(40, len(df) + 1, 40))
plt.legend()
plt.tight_layout()
plt.show()


# In[6]:


#35 PC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-35-1.xlsx'
df = pd.read_excel(file_path)

columns_to_analyze = [ 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']

def calculate_correlations(df, cycles, target='Capacity'):
    cycle_data = df[df['cycle number'] <= cycles]
    correlations = cycle_data[columns_to_analyze + [target]].corr()[target].drop(target)
    return abs(correlations)

correlations_1 = {}
for cycle in range(10, len(df)+1,10):
    correlations_1[cycle] = calculate_correlations(df, cycle)

average_correlations = np.mean([list(correlations_1[cycle].values) for cycle in range(10, len(df)+1,10)], axis=1)

plt.figure(figsize=(10, 6),dpi=600)

for feature in columns_to_analyze:
    corr_values = [correlations_1[cycle][feature] for cycle in range(10, len(df)+1,10)]
    plt.plot(range(10, len(df)+1,10), corr_values, marker='o', linestyle='-', label=feature)

plt.plot(range(10, len(df)+1,10), average_correlations, marker='o', linestyle='-', color='black', label='Average')

plt.title('Correlations with Capacity (Different Features) CS35')
plt.xlabel('Cycles')
plt.ylabel('Correlation')
plt.xticks(range(40, len(df) + 1, 40))
plt.legend()
plt.tight_layout()
plt.show()


# In[1]:


#1-W Distance
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

file_path_1 = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-33-0_5.xlsx'
file_path_2 = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-35-1.xlsx'

df1 = pd.read_excel(file_path_1).iloc[4:900, :]
df2 = pd.read_excel(file_path_2).iloc[4:900, :]

common_columns = list(set(df1.columns) & set(df2.columns))
columns_to_analyze=[ 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']

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

max_cycles = min(df1['cycle number'].max(), df2['cycle number'].max())
cycle_ranges = range(10, max_cycles + 1, 10)

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


# In[7]:


# Write to CALCE PC TC excel
import pandas as pd
import numpy as np

df_wasserstein = pd.DataFrame(wasserstein_distances)

correlations_df = pd.DataFrame(correlations)
correlations_df_1 = pd.DataFrame(correlations_1)

correlations_df = correlations_df.T
df_correlation = pd.DataFrame(correlations_df)
correlations_df_1 = correlations_df_1.T
df_correlation_1 = pd.DataFrame(correlations_df_1)

file_path = 'D:\\桌面上的文件\\CALCE_CS.xlsx'
with pd.ExcelWriter(file_path) as writer:
    df_wasserstein.to_excel(writer, sheet_name='TC_Trend', index=False)
    df_correlation.to_excel(writer, sheet_name='PC_Trend', index=False)
    df_correlation_1.to_excel(writer, sheet_name='PC_Trend_1', index=False)


print(f"数据已成功写入 {file_path}")


# In[5]:


#PC TC Trend
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\CALCE_CS.xlsx'  
df_TC = pd.read_excel(file_path, sheet_name='TC_Trend')  
df_PC = pd.read_excel(file_path, sheet_name='PC_Trend')  

mean_wasserstein = df_TC.mean(axis=1)
mean_correlation = df_PC.mean(axis=1)

fig, ax1 = plt.subplots(figsize=(4, 2), dpi=600)

color = '#599CB4'  
ax1.set_xlabel('Periods')
ax1.set_ylabel('Average Wasserstein Trend', color=color)
ax1.plot(mean_wasserstein[:21], marker='o', markersize=7,linestyle='-', linewidth=4,color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.8, 1)  

ax2 = ax1.twinx()  
color = '#C25759' 
ax2.set_ylabel('Average Correlation Trend', color=color)
ax2.plot(mean_correlation[:21], marker='o', markersize=10,linestyle='-', linewidth=5,color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.2, 1)  
plt.grid(False)
plt.show()


# In[1]:


#LSTM 35 is source domain, 33 is target domain
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

file_path_5 = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-35-1.xlsx'

selected_columns = [ 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[4:867]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_5 = scaler.fit_transform(Feature_5)

scaledFeature_5 = pd.DataFrame(data=scaledFeature_5)

print(scaledFeature_5.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_5 = time_series_to_supervised(scaledFeature_5,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
cap=(pd.read_excel(file_path_5).iloc[4:867,6]).values.reshape(-1, 1)
scaledCapacity_5 = scaler.fit_transform(cap)

n_steps_in =3 
n_steps_out=1
processedCapacity_5 = time_series_to_supervised(scaledCapacity_5,n_steps_in,n_steps_out)

data_x5 = processedFeature_5.loc[:,'0(t-3)':'4(t-1)']
data_y5=processedCapacity_5.loc[:,'0']

train_X5=data_x5.values[:867]
train_y5=data_y5[:867]
train_X5 = train_X5.reshape((train_X5.shape[0], n_steps_in, 5))


# In[2]:


# Train source model
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
#print(f"模型运行时间：{elapsed_time} 秒")


# In[53]:


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

file_path_5 = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-35-1.xlsx'  
selected_columns = [ 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[4:867]
file_path_6 = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-33-0_5.xlsx'  
Feature_6 = pd.read_excel(file_path_6, usecols=selected_columns).iloc[4:855]

capacity_5=pd.read_excel(file_path_5).iloc[4:867,6]
capacity_6=pd.read_excel(file_path_6).iloc[4:855,6]
cap_6=(capacity_6).values.reshape(-1, 1)
n_steps_in=3
n_steps_out = 1

np.random.seed(42)
tf.random.set_seed(42)

prediction_capability=[]
for train_samples in range(10,201,10):

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
    
    data_x6 = processedFeature_6.loc[:,'0(t-3)':'4(t-1)']
    data_y6=processedCapacity_6.loc[:,'0']
    data_y6=data_y6.values.reshape(-1,1)
   
    train_X6=data_x6.values[:train_samples]
    test_X6=data_x6.values[train_samples:]
    train_y6=data_y6[:train_samples]
    test_y6=data_y6[train_samples:]
    train_X6 = train_X6.reshape((train_X6.shape[0], n_steps_in, 5))
    test_X6 = test_X6.reshape((test_X6.shape[0], n_steps_in, 5))
   
    for layer in model.layers[:2]:  
        layer.trainable = False

    input_layer = Input(shape=(train_X6.shape[1], train_X6.shape[2]))
    lstm_output_1 = model.layers[0](input_layer)  
    lstm_output_2 = model.layers[1](lstm_output_1)  
    new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)
    #new_dense_1 = model.layers[2](lstm_output_2)
    new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)
    #new_output_layer = model.layers[3](new_dense_1)

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


# In[60]:


# CALCE Acc
import pandas as pd

prediction_df = pd.DataFrame(prediction_capability, columns=["CS35-1 to CS33-0.5"])

file_path_output = "D:\桌面上的文件\CALCE_Acc.xlsx"

prediction_df.to_excel(file_path_output, index=False)


# In[64]:


#Acc plot
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'D:/桌面上的文件/CALCE_Acc.xlsx'  
df = pd.read_excel(file_path)

column_data = df['CS35-1 to CS33-0.5']

plt.figure(figsize=(5, 2), dpi=600)

for i in range(len(column_data)):
    if i == 0:  
        plt.bar(i, column_data[i], width=0.8, color='#898988')
    else:
        plt.bar(i, column_data[i], width=0.8, color='#d7d7d7')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.grid(False)
plt.ylim(0.8,0.9)
plt.show()


# In[ ]:


#True-Pre Curve
#LSTM 35 is source domain, 33 is target domain
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

file_path_5 = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-35-1.xlsx'

selected_columns = [ 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[4:867]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_5 = scaler.fit_transform(Feature_5)

scaledFeature_5 = pd.DataFrame(data=scaledFeature_5)

print(scaledFeature_5.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_5 = time_series_to_supervised(scaledFeature_5,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
cap=(pd.read_excel(file_path_5).iloc[4:867,6]).values.reshape(-1, 1)
scaledCapacity_5 = scaler.fit_transform(cap)

n_steps_in =3 
n_steps_out=1
processedCapacity_5 = time_series_to_supervised(scaledCapacity_5,n_steps_in,n_steps_out)

data_x5 = processedFeature_5.loc[:,'0(t-3)':'4(t-1)']
data_y5=processedCapacity_5.loc[:,'0']

train_X5=data_x5.values[:867]
train_y5=data_y5[:867]
train_X5 = train_X5.reshape((train_X5.shape[0], n_steps_in, 5))


# In[ ]:


# Train source model
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
#print(f"模型运行时间：{elapsed_time} 秒")


# In[5]:


#Validate Without Finetune
import matplotlib
train_samples=10

selected_columns = ['Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']
file_path_6 = r'D:\桌面上的文件\Data Sufficiency\CALCE_CS\Features_lb_CS2-33-0_5.xlsx'  
Feature_6 = pd.read_excel(file_path_6, usecols=selected_columns).iloc[:867]


scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_6 = scaler.fit_transform(Feature_6)

scaledFeature_6 = pd.DataFrame(data=scaledFeature_6)

n_steps_in =3 
n_steps_out=1
processedFeature_6 = time_series_to_supervised(scaledFeature_6,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
cap_6=(pd.read_excel(file_path_6).iloc[:867,-1]).values.reshape(-1, 1)
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
plt.plot(pd.read_excel(file_path_6).iloc[:867,-1], label='True')
x_range = range(train_samples, train_samples+ len(inv_forecast_y6))
plt.plot(x_range,inv_forecast_y6,marker='.',label='LSTM',linestyle=None,markersize=5) 
plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[6]:


#With Finetune
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
plt.plot(pd.read_excel(file_path_6).iloc[:867,-1], label='True')
x_range = range(train_samples, train_samples+len(inv_forecast_y6t))
plt.plot(x_range,inv_forecast_y6t,marker='.',label='LSTM+Fine-tune',linestyle=None,markersize=5)

plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel(file_path_6)
initial_capacity = data.iloc[0, -1]  
threshold_capacity = 0.95 * initial_capacity 

plt.figure(figsize=(4, 3),dpi=600)
plt.plot(data.iloc[:867, -1], label='Target_True',linewidth=5,color=plt.cm.Blues(0.8))
plt.plot(pd.read_excel(file_path_5).iloc[4:867,6], label='Source',linewidth=5,color=plt.cm.Reds(0.8))

x_range = range(train_samples, train_samples + len(inv_forecast_y6t))
plt.plot(x_range, inv_forecast_y6t, label='Target_Pre', linestyle=None, linewidth=5,color=plt.cm.Greens(0.8))

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
plt.title('CALCE-CS233')
plt.show()

print(f"真实容量曲线与 0.8 初始容量的交点对应的循环数: {true_x_intersection}")
print(f"预测容量曲线与 0.8 初始容量的交点对应的循环数: {pred_x_intersection}")

