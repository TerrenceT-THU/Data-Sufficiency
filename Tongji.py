#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

file_path = r'D:\桌面上的文件\Data Sufficiency\Tongji\Data\CY25-05_1-#1.csv'  
data = pd.read_csv(file_path)

cycle_numbers = data['cycle number'].unique()
cmap = cm.get_cmap('Blues', len(cycle_numbers))
colors = cmap(np.linspace(0.2, 1, len(cycle_numbers)))
cmap = cm.get_cmap('Blues', len(cycle_numbers))
colors = cmap(np.linspace(0.2, 1, len(cycle_numbers)))
plt.figure(figsize=(4, 3),dpi=600)

for idx, (cycle_num, cycle_data) in enumerate(data.groupby('cycle number')):
    filtered_data = cycle_data[cycle_data['control/mA'] == 1750]
    if not filtered_data.empty:
        charging_voltage = filtered_data['Ecell/V'].values
        charging_voltage_time = filtered_data['Q charge/mA.h'].values
        plt.plot(charging_voltage_time, charging_voltage, label=f'Cycle {cycle_num}', linewidth=1,color=colors[idx])

sm = cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=cycle_numbers.min(), vmax=cycle_numbers.max()))
plt.ylim(2,4.25)
plt.show()


# In[8]:


#TJU-capacity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import zscore

def extract_capacity(file_path):
    data = pd.read_csv(file_path)
    capacities = []
    for cycle_num, cycle_data in data.groupby('cycle number'):
        filtered_data = cycle_data[(cycle_data['control/mA'] == 1750)] 
        if not filtered_data.empty:
            capacity = cycle_data['Q discharge/mA.h'].iloc[-1]  
            capacities.append(capacity)    
    return capacities

def smooth_capacity(capacities, window_size=5, poly_order=2):
    smoothed_capacities = savgol_filter(capacities, window_size, poly_order)
    return smoothed_capacities

def remove_outliers(data, threshold=2):
    z_scores = zscore(data)
    return [x for i, x in enumerate(data) if abs(z_scores[i]) <= threshold]

cy25_files = [f'D:\桌面上的文件\Data Sufficiency\Tongji\Data\CY25-05_1-#{i}.csv' for i in range(1, 20)]
cy25_capacities = [extract_capacity(file) for file in cy25_files]

cy45_files = [f'D:\桌面上的文件\Data Sufficiency\Tongji\Data\CY45-05_1-#{i}.csv' for i in range(1, 29)]
cy45_capacities = [extract_capacity(file) for file in cy45_files]


plt.figure(figsize=(4, 3),dpi=600)

for i, capacities in enumerate(cy45_capacities):
    filtered_capacities = remove_outliers(capacities)
    smoothed = smooth_capacity(filtered_capacities)  
    plt.plot(range(1, len(smoothed) + 1), smoothed, label=f'45°C - Battery {i+1}', marker='o',markersize=3,color=plt.cm.Blues(0.8))

for i, capacities in enumerate(cy25_capacities):
    filtered_capacities = remove_outliers(capacities)
    smoothed = smooth_capacity(filtered_capacities)  
    plt.plot(range(1, len(smoothed) + 1), smoothed, label=f'25°C - Battery {i+1}', marker='o',markersize=3,color=plt.cm.Reds(0.8))

plt.title('Capacity Decay vs Cycle ', fontsize=14)
plt.xlabel('Cycle', fontsize=12)
plt.ylabel('Capacity (mAh)', fontsize=12)
plt.tight_layout()
plt.show()


# In[2]:


# Feature Extraction CY45
import pandas as pd
import numpy as np
import os

file_dir = r'D:\桌面上的文件\Data Sufficiency\Tongji\Data'
file_template = 'CY45-05_1-#{}.csv'  
output_excel = r'D:\桌面上的文件\Features_Tongji-CY45-05.xlsx'  

with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    
    for i in range(1, 29):
        file_path = os.path.join(file_dir, file_template.format(i))        
        data = pd.read_csv(file_path)
        results = []

        for cycle_num, cycle_data in data.groupby('cycle number'):
            
            filtered_data = cycle_data[(cycle_data['control/mA'] == 1750)]
            filtered_data2 = cycle_data[(cycle_data['control/mA'] == 0) & (cycle_data['control/V'] != 0)]

            if not filtered_data.empty and not filtered_data2.empty:
               
                charging_voltage = filtered_data['Ecell/V'].values
                charging_voltage_time = filtered_data['time/s'].values
                charging_capacity = filtered_data2['Q charge/mA.h'].values
                charging_current_time = filtered_data2['time/s'].values

                Gra = np.diff(charging_voltage) / np.diff(charging_voltage_time)
                Vg = np.mean(Gra)
                Q = charging_capacity[-1]
                RL = (charging_voltage[-1] - charging_voltage[0]) / 1.75
                T_CCCV = charging_voltage_time[-1] - charging_voltage_time[0]
                T_CVCA = charging_current_time[-1] - charging_current_time[0]
                Capacity = cycle_data['Q discharge/mA.h'].iloc[-1]

                results.append([cycle_num, Vg, Q, RL, T_CCCV, T_CVCA, Capacity])

        result_df = pd.DataFrame(results, columns=['cycle number', 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA', 'Capacity'])

        sheet_name = f'#{i}'
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)

print("特征提取完成，并已保存到不同的Sheet中")


# In[3]:


# Feature Filter CY45
import pandas as pd
import numpy as np
from scipy.signal import medfilt
import os

def apply_median_filter(series, kernel_size=3):
    return medfilt(series, kernel_size=kernel_size)

input_file = r'D:\桌面上的文件\Features_Tongji-CY45-05.xlsx'
output_file = r'D:\桌面上的文件\Features_lb_Tongji-CY45-05.xlsx'

sheet_names = pd.ExcelFile(input_file).sheet_names

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
   
    for sheet_name in sheet_names:      
        df = pd.read_excel(input_file, sheet_name=sheet_name)      
        filtered_df = df.copy()

        for column in df.columns:            
            if pd.api.types.is_numeric_dtype(df[column]):
                filtered_df[column] = apply_median_filter(df[column].values, kernel_size=3)
    
        filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)

print("所有sheet的滤波处理已完成，并已保存到新文件中")


# In[7]:


# Feature Extraction CY35
import pandas as pd
import numpy as np
import os

file_dir = r'D:\桌面上的文件\Data Sufficiency\Tongji\Data'
file_template = 'CY35-05_1-#{}.csv'  
output_excel = r'D:\桌面上的文件\Features_Tongji-CY35-05.xlsx'  

with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    for i in range(1, 4):
        file_path = os.path.join(file_dir, file_template.format(i))
        data = pd.read_csv(file_path)
        results = []

        for cycle_num, cycle_data in data.groupby('cycle number'):
           
            filtered_data = cycle_data[(cycle_data['control/mA'] == 1750)]
            filtered_data2 = cycle_data[(cycle_data['control/mA'] == 0) & (cycle_data['control/V'] != 0)]

            if not filtered_data.empty and not filtered_data2.empty:
                
                charging_voltage = filtered_data['Ecell/V'].values
                charging_voltage_time = filtered_data['time/s'].values
                charging_capacity = filtered_data2['Q charge/mA.h'].values
                charging_current_time = filtered_data2['time/s'].values

                Gra = np.diff(charging_voltage) / np.diff(charging_voltage_time)
                Vg = np.mean(Gra)
                Q = charging_capacity[-1]
                RL = (charging_voltage[-1] - charging_voltage[0]) / 1.75
                T_CCCV = charging_voltage_time[-1] - charging_voltage_time[0]
                T_CVCA = charging_current_time[-1] - charging_current_time[0]
                Capacity = cycle_data['Q discharge/mA.h'].iloc[-1]

                results.append([cycle_num, Vg, Q, RL, T_CCCV, T_CVCA, Capacity])

        result_df = pd.DataFrame(results, columns=['cycle number', 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA', 'Capacity'])

        sheet_name = f'#{i}'
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)

print("特征提取完成，并已保存到不同的Sheet中")


# In[8]:


# Feature Filter CY35
import pandas as pd
import numpy as np
from scipy.signal import medfilt
import os

def apply_median_filter(series, kernel_size=3):
    return medfilt(series, kernel_size=kernel_size)

input_file = r'D:\桌面上的文件\Features_Tongji-CY35-05.xlsx'
output_file = r'D:\桌面上的文件\Features_lb_Tongji-CY35-05.xlsx'

sheet_names = pd.ExcelFile(input_file).sheet_names

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    for sheet_name in sheet_names:
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        filtered_df = df.copy()

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                filtered_df[column] = apply_median_filter(df[column].values, kernel_size=3)

        filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)

print("所有sheet的滤波处理已完成，并已保存到新文件中")


# In[10]:


# Feature Extraction CY25
import pandas as pd
import numpy as np
import os

file_dir = r'D:\桌面上的文件\Data Sufficiency\Tongji\Data'
file_template = 'CY25-05_1-#{}.csv'
output_excel = r'D:\桌面上的文件\Features_Tongji-CY25-05.xlsx'  

with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    for i in range(1, 20):
        file_path = os.path.join(file_dir, file_template.format(i))
        data = pd.read_csv(file_path)
        results = []

        for cycle_num, cycle_data in data.groupby('cycle number'):
           
            filtered_data = cycle_data[(cycle_data['control/mA'] == 1750)]
            filtered_data2 = cycle_data[(cycle_data['control/mA'] == 0) & (cycle_data['control/V'] != 0)]

            if not filtered_data.empty and not filtered_data2.empty:

                charging_voltage = filtered_data['Ecell/V'].values
                charging_voltage_time = filtered_data['time/s'].values
                charging_capacity = filtered_data2['Q charge/mA.h'].values
                charging_current_time = filtered_data2['time/s'].values

                Gra = np.diff(charging_voltage) / np.diff(charging_voltage_time)
                Vg = np.mean(Gra)
                Q = charging_capacity[-1]
                RL = (charging_voltage[-1] - charging_voltage[0]) / 1.75
                T_CCCV = charging_voltage_time[-1] - charging_voltage_time[0]
                T_CVCA = charging_current_time[-1] - charging_current_time[0]
                Capacity = cycle_data['Q discharge/mA.h'].iloc[-1]
                results.append([cycle_num, Vg, Q, RL, T_CCCV, T_CVCA, Capacity])
        result_df = pd.DataFrame(results, columns=['cycle number', 'Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA', 'Capacity'])
        sheet_name = f'#{i}'
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)

print("特征提取完成，并已保存到不同的Sheet中")


# In[11]:


# Feature Filter CY25
import pandas as pd
import numpy as np
from scipy.signal import medfilt
import os

def apply_median_filter(series, kernel_size=3):
    return medfilt(series, kernel_size=kernel_size)
input_file = r'D:\桌面上的文件\Features_Tongji-CY25-05.xlsx'
output_file = r'D:\桌面上的文件\Features_lb_Tongji-CY25-05.xlsx'
sheet_names = pd.ExcelFile(input_file).sheet_names

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    for sheet_name in sheet_names:
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        filtered_df = df.copy()
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                filtered_df[column] = apply_median_filter(df[column].values, kernel_size=3)
        filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)

print("所有sheet的滤波处理已完成，并已保存到新文件中")


# In[19]:


# Tongji CY45 PC  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\Features_lb_Tongji-CY45-05.xlsx'  
battery_sheet_CY45 = [f'#{i}' for i in range(1, 29)]
periods = range(10, 201, 10)
all_correlations = []

for sheet_name in battery_sheet_CY45:
    Feature_CY45 = pd.read_excel(file_path, sheet_name=sheet_name).iloc[2:, 1:6]  
    dis_cap = pd.read_excel(file_path, sheet_name=sheet_name).iloc[2:, 6] 
    correlation_trend_CY45 = []
 
    for col, feature_data in Feature_CY45.items():
        correlations_CY45 = []  

        for seg in periods:           
            feature_slice = feature_data.iloc[:seg]
            capacity_slice = dis_cap.iloc[:seg]  
            correlation = feature_slice.corr(capacity_slice)
            correlations_CY45.append(abs(correlation))

        correlation_trend_CY45.append(correlations_CY45)

    all_correlations.append(np.array(correlation_trend_CY45))

all_correlations = np.array(all_correlations)
average_correlations_CY45 = np.mean(all_correlations, axis=0)
average_correlations_overall_CY45 = np.mean(average_correlations_CY45, axis=0)

plt.figure(figsize=(5, 3))
for i in range(average_correlations.shape[0]):
    plt.plot(periods, average_correlations_CY45[i], label=f'Feature {i+1}')

plt.plot(periods, average_correlations_overall_CY45, 'o-', color='black', label='Average')
plt.xlabel('Cycles')
plt.ylabel('PC (Correlation)')
plt.title('Average Correlation Trend of Features with Capacity for CY45')
plt.legend()
plt.grid(True)
plt.show()


# In[9]:


# Tongji CY35 PC  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\Features_lb_Tongji-CY35-05.xlsx'  
battery_sheet_CY35 = [f'#{i}' for i in range(1, 4)]
periods = range(10, 401, 10)
all_correlations = []

for sheet_name in battery_sheet_CY35:
    Feature_CY35 = pd.read_excel(file_path, sheet_name=sheet_name).iloc[2:, 1:6]  
    dis_cap = pd.read_excel(file_path, sheet_name=sheet_name).iloc[2:, 6]    
    correlation_trend_CY35 = []
    
    for col, feature_data in Feature_CY35.items():
        correlations_CY35 = [] 

        for seg in periods:
            feature_slice = feature_data.iloc[:seg]
            capacity_slice = dis_cap.iloc[:seg]  
            correlation = feature_slice.corr(capacity_slice)
            correlations_CY35.append(abs(correlation))
        correlation_trend_CY35.append(correlations_CY35)
    all_correlations.append(np.array(correlation_trend_CY35))
all_correlations = np.array(all_correlations)

average_correlations = np.mean(all_correlations, axis=0)
average_correlations_overall = np.mean(average_correlations, axis=0)

plt.figure(figsize=(5, 3))
for i in range(average_correlations.shape[0]):
    plt.plot(periods, average_correlations[i], label=f'Feature {i+1}')

plt.plot(periods, average_correlations_overall, 'o-', color='black', label='Average')
plt.xlabel('Cycles')
plt.ylabel('PC (Correlation)')
plt.title('Average Correlation Trend of Features with Capacity for CY35')
plt.legend()
plt.grid(True)
plt.show()


# In[18]:


# Tongji CY25 PC 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\Features_lb_Tongji-CY25-05.xlsx'  
battery_sheet_CY25 = [f'#{i}' for i in range(1, 20)]

periods = range(10, 201, 10)
all_correlations = []

for sheet_name in battery_sheet_CY25:
    Feature_CY25 = pd.read_excel(file_path, sheet_name=sheet_name).iloc[3:, 1:6]  
    dis_cap = pd.read_excel(file_path, sheet_name=sheet_name).iloc[3:, 6]  
    correlation_trend_CY25 = []
    for col, feature_data in Feature_CY25.items():
        correlations_CY25 = []  

        for seg in periods:
            feature_slice = feature_data.iloc[:seg]
            capacity_slice = dis_cap.iloc[:seg]  
            correlation = feature_slice.corr(capacity_slice)
            correlations_CY25.append(abs(correlation))
        correlation_trend_CY25.append(correlations_CY25)
    all_correlations.append(np.array(correlation_trend_CY25))

all_correlations = np.array(all_correlations)
average_correlations = np.mean(all_correlations, axis=0)
average_correlations_overall = np.mean(average_correlations, axis=0)

plt.figure(figsize=(5, 3))
for i in range(average_correlations.shape[0]):
    plt.plot(periods, average_correlations[i], label=f'Feature {i+1}')

plt.plot(periods, average_correlations_overall, 'o-', color='black', label='Average')
plt.xlabel('Cycles')
plt.ylabel('PC (Correlation)')
plt.title('Average Correlation Trend of Features with Capacity for CY25')
plt.legend()
plt.grid(True)
plt.show()


# In[20]:


# Tongji TC CY25 to CY45 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

file_path_Batch1 = r'D:\桌面上的文件\Features_lb_Tongji-CY25-05.xlsx'
file_path_Batch2 = r'D:\桌面上的文件\Features_lb_Tongji-CY45-05.xlsx'

battery_sheet_Batch1 = [f'#{i}' for i in range(1, 20)]
battery_sheet_Batch2 = [f'#{i}' for i in range(1, 29)]

feature_names = ['Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']

periods = list(range(10, 201, 10))

wasserstein_distances_per_feature = {feature: [] for feature in feature_names}

for sheet_Batch1 in battery_sheet_Batch1:
    for sheet_Batch2 in battery_sheet_Batch2:

        df_battery_Batch1 = pd.read_excel(file_path_Batch1, sheet_name=sheet_Batch1).iloc[2:, 1:6] 
        df_battery_Batch2 = pd.read_excel(file_path_Batch2, sheet_name=sheet_Batch2).iloc[2:, 1:6] 

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

plt.title('1-Wasserstein Distance Between CY45 and CY25 for Each Feature')
plt.xlabel('Cycles')
plt.ylabel('Average Transferable Capability (1 - W Distance)')
plt.grid(True)
plt.legend(title='Features')
plt.show()


# In[21]:


import pandas as pd
import numpy as np

df_wasserstein = pd.DataFrame(average_wasserstein_distances_per_feature)
df_correlation = pd.DataFrame(average_correlations.T, columns=[f'Feature_{i+1}' for i in range(5)])
df_correlation_CY45 = pd.DataFrame(average_correlations_CY45.T, columns=[f'Feature_{i+1}' for i in range(5)])

file_path = 'D:\\桌面上的文件\\Tongji.xlsx'
with pd.ExcelWriter(file_path) as writer:
    df_wasserstein.to_excel(writer, sheet_name='TC_Trend', index=False)
    df_correlation.to_excel(writer, sheet_name='PC_Trend', index=False)
    df_correlation_CY45.to_excel(writer, sheet_name='PC_Trend_CY45', index=False)

print(f"数据已成功写入 {file_path}")


# In[22]:


#PC TC Trend
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\Tongji.xlsx'  
df_TC = pd.read_excel(file_path, sheet_name='TC_Trend')  
df_PC = pd.read_excel(file_path, sheet_name='PC_Trend_CY45')  

mean_wasserstein = df_TC.mean(axis=1)
mean_correlation = df_PC.mean(axis=1)

fig, ax1 = plt.subplots(figsize=(4, 2), dpi=600)

color = '#599CB4'  
ax1.set_xlabel('Periods')
ax1.set_ylabel('Average Wasserstein Trend', color=color)
ax1.plot(mean_wasserstein, marker='o', markersize=10,linestyle='-', linewidth=5,color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.7, 1) 

ax2 = ax1.twinx()  
color = '#C25759'  
ax2.set_ylabel('Average Correlation Trend', color=color)
ax2.plot(mean_correlation, marker='o', markersize=10,linestyle='-', linewidth=5,color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.8, 1)  
plt.grid(False)
plt.show()


# In[1]:


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

file_path_CY45 = r'D:\桌面上的文件\Data Sufficiency\Tongji\Features_lb_Tongji-CY25-05.xlsx'
selected_columns = ['Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']
Feature_CY45 = pd.read_excel(file_path_CY45, sheet_name='#9', usecols=selected_columns).iloc[3:800,0:5]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_CY45 = scaler.fit_transform(Feature_CY45)
scaledFeature_CY45 = pd.DataFrame(data=scaledFeature_CY45)
print(scaledFeature_CY45.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_CY45 = time_series_to_supervised(scaledFeature_CY45,n_steps_in,n_steps_out)
scaler = MinMaxScaler(feature_range=(0, 1))
cap=(pd.read_excel(file_path_CY45,sheet_name='#9').iloc[3:800,6]).values.reshape(-1, 1)
scaledCapacity_CY45 = scaler.fit_transform(cap)

n_steps_in =3 
n_steps_out=1
processedCapacity_CY45 = time_series_to_supervised(scaledCapacity_CY45,n_steps_in,n_steps_out)
data_xCY45 = processedFeature_CY45.loc[:,'0(t-3)':'4(t-1)']
data_yCY45=processedCapacity_CY45.loc[:,'0']
train_XCY45=data_xCY45.values[:800]
train_yCY45=data_yCY45[:800]
train_XCY45 = train_XCY45.reshape((train_XCY45.shape[0], n_steps_in, 5))


# In[2]:


# Train Source Model
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
model = Sequential()#序列类型神经网络
model.add(LSTM(96, return_sequences=True, input_shape=(train_XCY45.shape[1], train_XCY45.shape[2]))) # 返回完整的输出序列，input_shape定义输出数据的形状
model.add(LSTM(64, return_sequences=False))#只返回输出序列的最后一步

model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

history = model.fit(train_XCY45, train_yCY45, epochs=30, batch_size=64,verbose=2, shuffle=False)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"模型运行时间：{elapsed_time} 秒")


# In[3]:


# T25 to T45，look at prediction accuracy in different fine-tuning cycles automaticly 
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

file_path_CY25 = r'D:\桌面上的文件\Features_lb_Tongji-CY45-05.xlsx'  
selected_columns = ['Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']
Feature_CY25 = pd.read_excel(file_path_CY25, sheet_name='#12',usecols=selected_columns).iloc[3:800,0:5]
capacity_CY25=pd.read_excel(file_path_CY25,sheet_name='#12').iloc[3:800,6]
cap_CY25=capacity_CY25.values.reshape(-1, 1)
n_steps_in=3
n_steps_out = 1

np.random.seed(42)
tf.random.set_seed(42)

prediction_capability=[]
for train_samples in range(10,201,10):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaledFeature_CY25 = scaler.fit_transform(Feature_CY25)
    scaledFeature_CY25 = pd.DataFrame(data=scaledFeature_CY25)
    n_steps_in =3 
    n_steps_out=1
    processedFeature_CY25 = time_series_to_supervised(scaledFeature_CY25,n_steps_in,n_steps_out)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledCapacity_CY25 = scaler.fit_transform(cap_CY25)

    n_steps_in =3 
    n_steps_out=1
    processedCapacity_CY25 = time_series_to_supervised(scaledCapacity_CY25,n_steps_in,n_steps_out)
    data_xCY25 = processedFeature_CY25.loc[:,'0(t-3)':'4(t-1)']
    data_yCY25=processedCapacity_CY25.loc[:,'0']
    data_yCY25=data_yCY25.values.reshape(-1,1)
    train_XCY25=data_xCY25.values[:train_samples]
    test_XCY25=data_xCY25.values[train_samples:]
    train_yCY25=data_yCY25[:train_samples]
    test_yCY25=data_yCY25[train_samples:]
    train_XCY25 = train_XCY25.reshape((train_XCY25.shape[0], n_steps_in, 5))
    test_XCY25 = test_XCY25.reshape((test_XCY25.shape[0], n_steps_in, 5))
 
    for layer in model.layers[:2]:  
        layer.trainable = False

    input_layer = Input(shape=(train_XCY25.shape[1], train_XCY25.shape[2]))
    lstm_output_1 = model.layers[0](input_layer)  
    lstm_output_2 = model.layers[1](lstm_output_1)  
    new_dense_1 = Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(lstm_output_2)
    new_output_layer = Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(new_dense_1)
   
    transfer_model = Model(inputs=input_layer, outputs=new_output_layer)
    transfer_model.compile(loss='mse', optimizer='adam')
    transfer_model.fit(train_XCY25, train_yCY25, epochs=50, batch_size=64, verbose=2, shuffle=False)

    yhat_CY25t= transfer_model.predict(test_XCY25)
    test_yCY25=test_yCY25.reshape(-1,1) 
  
    inv_forecast_yCY25t = scaler.inverse_transform(yhat_CY25t)
    inv_test_yCY25t = scaler.inverse_transform(test_yCY25)
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true))
    mape_CY25t = mape(inv_test_yCY25t, inv_forecast_yCY25t)
    print('Test MAPE: %.3f' % mape_CY25t)
    prediction_capability.append(1-mape_CY25t)
prediction_capability


# In[9]:


# Tongji Acc
import pandas as pd

prediction_df = pd.DataFrame(prediction_capability, columns=["25#9 to 45#12"])

file_path_output = r"D:\桌面上的文件\TongJi_Acc_Target45.xlsx"

prediction_df.to_excel(file_path_output, index=False)


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt

file_path = 'D:/桌面上的文件/TongJi_Acc_Target45.xlsx'  # 将 'your_file.xlsx' 替换为实际的文件路径
df = pd.read_excel(file_path)

column_data = df['25#9 to 45#12']
column_data = df['25#9 to 45#12']

plt.figure(figsize=(5, 2), dpi=600)

for i in range(len(column_data)):
    if i == 1:  
        plt.bar(i, column_data[i], width=0.8, color='#898988')
    else:
        plt.bar(i, column_data[i], width=0.8, color='#d7d7d7')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.grid(False)
plt.ylim(0.97,1)
plt.show()


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

file_path_CY45 = r'D:\桌面上的文件\Data Sufficiency\Tongji\Features_lb_Tongji-CY25-05.xlsx'
selected_columns = ['Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']
Feature_CY45 = pd.read_excel(file_path_CY45, sheet_name='#9', usecols=selected_columns).iloc[3:800,0:5]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_CY45 = scaler.fit_transform(Feature_CY45)
scaledFeature_CY45 = pd.DataFrame(data=scaledFeature_CY45)
print(scaledFeature_CY45.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_CY45 = time_series_to_supervised(scaledFeature_CY45,n_steps_in,n_steps_out)
scaler = MinMaxScaler(feature_range=(0, 1))
cap=(pd.read_excel(file_path_CY45,sheet_name='#9').iloc[3:800,6]).values.reshape(-1, 1)
scaledCapacity_CY45 = scaler.fit_transform(cap)

n_steps_in =3 
n_steps_out=1
processedCapacity_CY45 = time_series_to_supervised(scaledCapacity_CY45,n_steps_in,n_steps_out)
data_xCY45 = processedFeature_CY45.loc[:,'0(t-3)':'4(t-1)']
data_yCY45=processedCapacity_CY45.loc[:,'0']
train_XCY45=data_xCY45.values[:800]
train_yCY45=data_yCY45[:800]
train_XCY45 = train_XCY45.reshape((train_XCY45.shape[0], n_steps_in, 5))


# In[ ]:


# Train Source Model
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
model = Sequential()#序列类型神经网络
model.add(LSTM(96, return_sequences=True, input_shape=(train_XCY45.shape[1], train_XCY45.shape[2]))) # 返回完整的输出序列，input_shape定义输出数据的形状
model.add(LSTM(64, return_sequences=False))#只返回输出序列的最后一步

model.add(Dense(32, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))
model.add(Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42)))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

history = model.fit(train_XCY45, train_yCY45, epochs=30, batch_size=64,verbose=2, shuffle=False)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"模型运行时间：{elapsed_time} 秒")


# In[3]:


# T45 Validation no finetune
import matplotlib
train_samples=20

selected_columns = ['Vg', 'Q', 'RL', 'T_CCCV', 'T_CVCA']
file_path_6 = r'D:\桌面上的文件\Data Sufficiency\Tongji\Features_lb_Tongji-CY45-05.xlsx'  
Feature_6 = pd.read_excel(file_path_6,sheet_name='#12',usecols=selected_columns).iloc[3:800,0:5]
capacity_6=pd.read_excel(file_path_6,sheet_name='#12').iloc[3:800,6]
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
plt.plot(pd.read_excel(file_path_6, sheet_name='#12').iloc[:800,-1], label='True')
x_range = range(train_samples, train_samples+ len(inv_forecast_y6))
plt.plot(x_range,inv_forecast_y6,marker='.',label='LSTM',linestyle=None,markersize=5)

plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[4]:


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
plt.plot(pd.read_excel(file_path_6,sheet_name='#12').iloc[:800,-1], label='True')
x_range = range(train_samples, train_samples+len(inv_forecast_y6t))
plt.plot(x_range,inv_forecast_y6t,marker='.',label='LSTM+Fine-tune',linestyle=None,markersize=5)
plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel(file_path_6,sheet_name='#12')
initial_capacity = data.iloc[0, -1]  
threshold_capacity = 0.95 * initial_capacity  

plt.figure(figsize=(4, 3),dpi=600)
plt.plot(data.iloc[:800, -1], label='Target_True',linewidth=5,color=plt.cm.Blues(0.8))
plt.plot(pd.read_excel(file_path_CY45,sheet_name='#10').iloc[3:800,-1],label='Source',linewidth=5,color=plt.cm.Reds(0.8))#画源域

x_range = range(train_samples, train_samples + len(inv_forecast_y6t))
#plt.plot(x_range, inv_forecast_y6t, marker='.', label='LSTM+Fine-tune', linestyle=None, markersize=5)
plt.plot(inv_forecast_y6t,label='Target_Pre', linestyle=None, linewidth=5,color=plt.cm.Greens(0.8))

#plt.axvline(x=train_samples, color='gray', linestyle='--')  
#plt.axhline(y=threshold_capacity, color='red', linestyle='--', label='80% Capacity')

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
plt.title('TJU-T45#12')
#plt.legend(fontsize=12)
plt.ylim(initial_capacity*0.7,initial_capacity*1.05)
plt.show()

print(f"真实容量曲线与 0.8 初始容量的交点对应的循环数: {true_x_intersection}")
print(f"预测容量曲线与 0.8 初始容量的交点对应的循环数: {pred_x_intersection}")

