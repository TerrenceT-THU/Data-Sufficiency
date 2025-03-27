#!/usr/bin/env python
# coding: utf-8

# In[2]:


# voltage-SOC curve
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

df = pd.read_csv(r'D:\桌面上的文件\Data Sufficiency\MIT\Data\2017-05-12_6C-40per_3C_CH25.csv')

cycle_indices = df['Cycle_Index'].unique()[1:-20]

cmap = cm.get_cmap('Blues', len(cycle_indices))
colors = cmap(np.linspace(0, 1, len(cycle_indices)))

plt.figure(figsize=(4, 3), dpi=600)

voltage_at_08 = {}

for cycle in cycle_indices:
    
    cycle_data = df[df['Cycle_Index'] == cycle]

    step5_data = cycle_data[cycle_data['Step_Index'] == 5]
    step6_data = cycle_data[cycle_data['Step_Index'] == 6]
    
    if not step5_data.empty and not step6_data.empty:
        
        t5 = step5_data['Charge_Capacity'].values
        v5 = step5_data['Voltage'].values
        t6 = step6_data['Charge_Capacity'].values
        v6 = step6_data['Voltage'].values

        time_step5 = t5 - t5[0]
        time_step6 = t6 - t6[0] + time_step5[-1]
        time = np.concatenate([time_step5, time_step6])
        voltage = np.concatenate([v5, v6])

        if time.min() <= 0.8 <= time.max(): 
            interp_voltage = np.interp(0.78, time, voltage)
            voltage_at_08[cycle] = interp_voltage

if voltage_at_08:
    target_cycle = min(voltage_at_08, key=voltage_at_08.get)
    print(f"Cycle {target_cycle} has the minimum voltage at x=0.8 and will be removed.")

for idx, cycle in enumerate(cycle_indices):
    if cycle == target_cycle:  
        continue

    cycle_data = df[df['Cycle_Index'] == cycle]
    step5_data = cycle_data[cycle_data['Step_Index'] == 5]
    step6_data = cycle_data[cycle_data['Step_Index'] == 6]
    
    if not step5_data.empty and not step6_data.empty:
        
        t5 = step5_data['Charge_Capacity'].values
        v5 = step5_data['Voltage'].values
        t6 = step6_data['Charge_Capacity'].values
        v6 = step6_data['Voltage'].values

        time_step5 = t5 - t5[0]
        time_step6 = t6 - t6[0] + time_step5[-1]
        time = np.concatenate([time_step5, time_step6])
        voltage = np.concatenate([v5, v6])

        plt.plot(time, voltage, label=f'Cycle {cycle}', linewidth=1,color=colors[idx])

sm = cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=cycle_indices.min(), vmax=cycle_indices.max()))

plt.title('Voltage-Capacity for Steps 5 and 6')
plt.xlabel('Capacity')
plt.ylabel('Volatge')
plt.ylim(2,4.25)
plt.show()


# In[3]:


#MIT capacity

import pandas as pd
import matplotlib.pyplot as plt
import os

file1 = r'D:\桌面上的文件\Data Sufficiency\MIT\Filtered_Features_MIT.xlsx'
file2 = r'D:\桌面上的文件\Data Sufficiency\MIT\Filtered_Features_MIT_14.xlsx'

data1 = pd.read_excel(file1, header=None)  
capacity1 = data1.iloc[1:, 14]  

data2 = pd.read_excel(file2, header=None)  
capacity2 = data2.iloc[1:, 14]  

plt.figure(figsize=(4, 3), dpi=600)
plt.plot(capacity2, label='File 2', marker='o', markersize=3, color=plt.cm.Reds(0.8))
plt.plot(capacity1, label='File 1', marker='o', markersize=3, color=plt.cm.Blues(0.8))

plt.title('MIT', fontsize=10)
plt.xlabel('Cycle Index', fontsize=10)
plt.ylabel('Capacity (Ah)', fontsize=10)

plt.tight_layout()
plt.show()


# In[ ]:


# Feature extraction
import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\桌面上的文件\2017-05-12_5_4C-50per_3C_CH14.csv')

results = []

cycle_indices = df['Cycle_Index'].unique()

for cycle in cycle_indices:
   
    cycle_data = df[df['Cycle_Index'] == cycle]
    Discharge_cap=cycle_data['Discharge_Capacity'].iloc[-1]     
   
    step5_data = cycle_data[cycle_data['Step_Index'] == 5]
    if not step5_data.empty:
        X5 = step5_data['Data_Point'].values
        y5 = step5_data['Voltage'].values
        I5 = step5_data['Current'].values
        t5 = step5_data['Test_Time'].values
     
        Gra_5 = np.diff(y5) / np.diff(X5)
        Vg1=np.mean(Gra_5)
        RL1 = (y5[-1] - y5[0])/I5[-1]
        Q1 = step5_data['Charge_Capacity'].values[-1]
    else:
        Vg1, RL1, Q1 = np.nan, np.nan, np.nan

    step6_data = cycle_data[cycle_data['Step_Index'] == 6]
    if not step6_data.empty:
        X6 = step6_data['Data_Point'].values
        y6 = step6_data['Voltage'].values
        I6 = step6_data['Current'].values
        t6 = step6_data['Test_Time'].values
   
        Gra_6 = np.diff(y6) / np.diff(X6)
        Vg2=np.mean(Gra_6)
        RL2 = (y6[-1] - y6[0])/I6[-1]
        Q2 = step6_data['Charge_Capacity'].values[-1]
        
        RO= (y6[0] - y5[-1])/(I6[-1]-I5[-1])
        VC21=y6[0] - y5[-1]
        y6_max=max(y6[:10])
        y6_max_index = np.argmax(y6[:10])
        y6_min=min(y6[y6_max_index:])
        y6_min_index = np.argmin(y6[y6_max_index:]) + y6_max_index
        VD2=y6_max-y6_min
        tVD2=t6[y6_min_index] - t6[y6_max_index]
    step7_data = cycle_data[cycle_data['Step_Index'] == 7]
    if not step7_data.empty:
        y7 = step7_data['Voltage'].values
        t7 = step7_data['Test_Time'].values
        ReVC=y6[-1]-y7[0]
        ReVD=y7[0]-y7[-1]
        y7_threshold=0.8*y7[0]
        y7_threshold_index = (np.abs(y7 - y7_threshold)).argmin()
        tReVD=t7[y7_threshold_index]-t7[0]
              
    else:
        Vg2, RL2, Q2, RO = np.nan, np.nan, np.nan, np.nan

    results.append([cycle, Vg1, Vg2, Q1, Q2,RL1, RL2,RO,VC21,VD2,tVD2,ReVC,ReVD,tReVD,Discharge_cap])

result_df = pd.DataFrame(results, columns=['Cycle_Index', 'Vg1', 'Vg2', 'Q1', 'Q2', 'RL1', 'RL2', 'RO','VC21','VD2','tVD2','ReVC','ReVD','tReVD','Capacity'])

result_df.to_excel('E:\MIT\Features_MIT_14.xlsx', index=False)

print("特征提取完成，并已保存")


# In[ ]:


# Feature extraction CH42
import pandas as pd
import numpy as np
from scipy.integrate import simps

df = pd.read_csv(r'E:\MIT\2017-05-12_7C-40per_3_6C_CH42.csv')

results = []

cycle_indices = df['Cycle_Index'].unique()

for cycle in cycle_indices:
    
    cycle_data = df[df['Cycle_Index'] == cycle]
    Discharge_cap=cycle_data['Discharge_Capacity'].iloc[-1]   
    Q1=cycle_data['Charge_Capacity'].iloc[-1] 
    Q2=cycle_data['Charge_Capacity'].iloc[-1] 
    step5_data = cycle_data[cycle_data['Step_Index'] == 5]
    if not step5_data.empty:
        X5 = step5_data['Data_Point'].values
        y5 = step5_data['Voltage'].values
        I5 = step5_data['Current'].values
        t5 = step5_data['Test_Time'].values
        Gra_5 = np.diff(y5[50:120]) / np.diff(t5[50:120])
        Vg1=np.mean(Gra_5)
        RL1 = (y5[-1] - y5[0])/I5[-1]
    else:
        Vg1, RL1, Q1 = np.nan, np.nan, np.nan
    
    
    step6_data = cycle_data[cycle_data['Step_Index'] == 6]
    if not step6_data.empty:
        X6 = step6_data['Data_Point'].values
        y6 = step6_data['Voltage'].values
        I6 = step6_data['Current'].values
        t6 = step6_data['Test_Time'].values
        Gra_6 = np.diff(y6[15:]) / np.diff(t6[15:])
        Vg2=np.mean(Gra_6)
        RL2 = (y6[-1] - y6[0])/I6[-1]
        RO= (y6[0] - y5[-1])/(I6[-1]-I5[-1])
        VC21=y6[0] - y5[-1]
        y6_max=max(y6[:10])
        y6_max_index = np.argmax(y6[:10])
        y6_min=min(y6[y6_max_index:])
        y6_min_index = np.argmin(y6[y6_max_index:]) + y6_max_index
        VD2=y6_max-y6_min
        tVD2=t6[y6_min_index] - t6[y6_max_index]
    step7_data = cycle_data[cycle_data['Step_Index'] == 7]
    if not step7_data.empty:
        y7 = step7_data['Voltage'].values
        t7 = step7_data['Test_Time'].values
        ReVC=y6[-1]-y7[0]
        ReVD=y7[0]-y7[-1]
        y7_threshold=0.8*y7[0]
        y7_threshold_index = (np.abs(y7 - y7_threshold)).argmin()
        tReVD=t7[y7_threshold_index]-t7[0]
              
    else:
        Vg2, RL2, Q2, RO = np.nan, np.nan, np.nan, np.nan

    results.append([cycle, Vg1, Vg2, Q1, Q2,RL1, RL2,RO,VC21,VD2,tVD2,ReVC,ReVD,tReVD,Discharge_cap])

result_df = pd.DataFrame(results, columns=['Cycle_Index', 'Vg1', 'Vg2', 'Q1', 'Q2', 'RL1', 'RL2', 'RO','VC21','VD2','tVD2','ReVC','ReVD','tReVD','Capacity'])
result_df.to_excel('E:\MIT\Features_MIT.xlsx', index=False)
print("特征提取完成，并已保存")


# In[ ]:


#Feature filtering Filtered_Features_MIT
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

file_path = r'E:\MIT\Features_MIT.xlsx'
df = pd.read_excel(file_path)

columns_to_process = ['Vg1', 'Vg2', 'Q1', 'Q2', 'RL1', 'RL2', 'RO', 'VC21', 'VD2', 'tVD2', 'ReVC', 'ReVD', 'tReVD']

threshold = 0.5  

def handle_step_anomalies(data, threshold):
    diff = data.diff().abs()
    indices = diff > threshold
    for i in range(1, len(data)):
        if indices.iloc[i]:
            data.iloc[i] = (data.iloc[i-1] + data.iloc[i+1]) / 2 if i+1 < len(data) else data.iloc[i-1]
    return data

window_length = 15  
polyorder = 3      

for column in columns_to_process:
    if column in df.columns:
        
        df[column] = handle_step_anomalies(df[column], threshold)
       
        df[column] = savgol_filter(df[column], window_length, polyorder)

output_file_path = r'E:\MIT\Filtered_Features_MIT.xlsx'
df.to_excel(output_file_path, index=False)

print(f'数据处理完成，结果已保存至: {output_file_path}')


# In[ ]:


#Feature filtering Filtered_Features_MIT_14
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

file_path = r'E:\MIT\Features_MIT_14.xlsx'
df = pd.read_excel(file_path)

columns_to_process = ['Vg1', 'Vg2', 'Q1', 'Q2', 'RL1', 'RL2', 'RO', 'VC21', 'VD2', 'tVD2', 'ReVC', 'ReVD', 'tReVD']

threshold = 0.5  

def handle_step_anomalies(data, threshold):
    diff = data.diff().abs()
    indices = diff > threshold
    for i in range(1, len(data)):
        if indices.iloc[i]:
            data.iloc[i] = (data.iloc[i-1] + data.iloc[i+1]) / 2 if i+1 < len(data) else data.iloc[i-1]
    return data

window_length = 15  
polyorder = 3      

for column in columns_to_process:
    if column in df.columns:
    
        df[column] = handle_step_anomalies(df[column], threshold)
  
        df[column] = savgol_filter(df[column], window_length, polyorder)

output_file_path = r'E:\MIT\Filtered_Features_MIT_14.xlsx'
df.to_excel(output_file_path, index=False)

print(f'数据处理完成，结果已保存至: {output_file_path}')


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = r'E:\MIT\Filtered_Features_MIT.xlsx'
df = pd.read_excel(file_path)

# 定义需要计算相关性的列
columns_to_analyze = [ 'Vg2', 'RL1', 'RL2', 'RO', 'VC21', 'VD2', 'ReVC', 'ReVD', ]
columns_to_analyze = [ 'Vg1','Vg2', 'RL1', 'RL2', 'RO', 'VC21', 'VD2','ReVC', 'ReVD']
columns_to_analyze = [ 'Vg1','Vg2', 'Q1','Q2','RL1', 'RL2', 'RO', 'VC21', 'VD2','tVD2','ReVC', 'ReVD','tReVD']
#columns_to_analyze =['Vg2', 'RL2', 'VC21', 'VD2', 'ReVC', 'ReVD']
# 相关性计算函数
def calculate_correlations(df, cycles, target='Capacity'):
    cycle_data = df[df['Cycle_Index'] <= cycles]
    correlations = cycle_data[columns_to_analyze + [target]].corr()[target].drop(target)
    return abs(correlations)

# 计算不同周期的相关性
correlations_6C = {}
for cycle in range(20, len(df)+1,20):
    correlations_6C[cycle] = calculate_correlations(df, cycle)
    
# 计算所有特征的平均相关性
average_correlations = np.mean([list(correlations_6C[cycle].values) for cycle in range(20, len(df)+1,20)], axis=1)

# 绘制相关性的图表
plt.figure(figsize=(10, 6),dpi=600)

for feature in columns_to_analyze:
    corr_values = [correlations_6C[cycle][feature] for cycle in range(20, len(df)+1,20)]
    plt.plot(range(20, len(df)+1,20), corr_values, marker='o', linestyle='-', label=feature)

# 绘制所有特征的平均相关性曲线
plt.plot(range(20, len(df)+1,20), average_correlations, marker='o', linestyle='-', color='black', label='Average')

plt.title('Correlations with Capacity (Different Features) 6C')
plt.xlabel('Cycles')
plt.ylabel('Correlation')
plt.xticks(range(40, len(df) + 1, 40))
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = r'E:\MIT\Filtered_Features_MIT_14.xlsx'
df = pd.read_excel(file_path)

# 定义需要计算相关性的列
columns_to_analyze = [ 'Vg1','Vg2', 'RL1', 'RL2', 'RO', 'VC21', 'VD2','ReVC', 'ReVD']
columns_to_analyze = [ 'Vg1','Vg2', 'Q1','Q2','RL1', 'RL2', 'RO', 'VC21', 'VD2','tVD2','ReVC', 'ReVD','tReVD']
#columns_to_analyze =['Vg2', 'RL2', 'VC21', 'VD2', 'ReVC', 'ReVD']
#columns_to_analyze = [ 'Vg2',  'RL2', 'ReVC', 'ReVD']
# 相关性计算函数
def calculate_correlations(df, cycles, target='Capacity'):
    cycle_data = df[df['Cycle_Index'] <= cycles]
    correlations = cycle_data[columns_to_analyze + [target]].corr()[target].drop(target)
    return abs(correlations)

# 计算不同周期的相关性
correlations = {}
for cycle in range(20, len(df)+1,20):
    correlations[cycle] = calculate_correlations(df, cycle)
    
# 计算所有特征的平均相关性
average_correlations = np.mean([list(correlations[cycle].values) for cycle in range(20, len(df)+1,20)], axis=1)

# 绘制相关性的图表
plt.figure(figsize=(10, 6),dpi=600)

for feature in columns_to_analyze:
    corr_values = [correlations[cycle][feature] for cycle in range(20, len(df)+1,20)]
    plt.plot(range(20, len(df)+1,20), corr_values, marker='o', linestyle='-', label=feature)

# 绘制所有特征的平均相关性曲线
plt.plot(range(20, len(df)+1,20), average_correlations, marker='o', linestyle='-', color='black', label='Average')

plt.title('Correlations with Capacity (Different Features) 4C')
plt.xlabel('Cycles')
plt.ylabel('Correlation')
plt.xticks(range(40, len(df) + 1, 40))
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 读取Excel文件
file_path_1 = r'E:\MIT\Filtered_Features_MIT.xlsx'
file_path_2 = r'E:\MIT\Filtered_Features_MIT_14.xlsx'

df1 = pd.read_excel(file_path_1).iloc[:700, :]
df2 = pd.read_excel(file_path_2).iloc[:700, :]

# 定义需要计算1-Wasserstein距离的列，排除Q1和Q2
common_columns = list(set(df1.columns) & set(df2.columns))
#columns_to_analyze = [col for col in common_columns if col in ['Vg2', 'RL2', 'VC21', 'VD2', 'ReVC', 'ReVD']]
#columns_to_analyze = [col for col in common_columns if col in [ 'Vg1','Vg2', 'RL1', 'RL2', 'RO', 'VC21', 'VD2', 'tVD2','ReVC', 'ReVD', 'tReVD']]
columns_to_analyze=[ 'Vg1','Vg2', 'RL1', 'RL2', 'RO', 'VC21', 'VD2','ReVC', 'ReVD']
columns_to_analyze = [ 'Vg1','Vg2', 'Q1','Q2','RL1', 'RL2', 'RO', 'VC21', 'VD2','tVD2','ReVC', 'ReVD','tReVD']
#columns_to_analyze=[ 'Vg2',  'RL2',  'ReVC', 'ReVD']
# 归一化处理
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

df1[columns_to_analyze] = scaler1.fit_transform(df1[columns_to_analyze])
df2[columns_to_analyze] = scaler2.fit_transform(df2[columns_to_analyze])

# 计算1-Wasserstein距离函数
def calculate_wasserstein_distance(df1, df2, cycles, columns):
    distances = {}
    for col in columns:
        data1 = df1.iloc[:cycles][col].dropna()
        data2 = df2.iloc[:cycles][col].dropna()
        if not data1.empty and not data2.empty:
            distance = wasserstein_distance(data1, data2)
            distances[col] = 1 - distance
    return distances

# 定义周期范围
max_cycles = min(df1['Cycle_Index'].max(), df2['Cycle_Index'].max())
cycle_ranges = range(20, max_cycles + 1, 20)

# 计算不同周期范围下每个特征的1-Wasserstein距离
wasserstein_distances = {col: [] for col in columns_to_analyze}
average_distances = []

for cycle_range in cycle_ranges:
    distances = calculate_wasserstein_distance(df1, df2, cycle_range, columns_to_analyze)
    for col, distance in distances.items():
        wasserstein_distances[col].append(distance)
    # 计算当前周期范围内所有特征的平均值
    if distances:
        average_distance = np.mean(list(distances.values()))
        average_distances.append(average_distance)
    else:
        average_distances.append(0)

# 绘制1-Wasserstein距离的图表
plt.figure(figsize=(10, 6),dpi=600)

for col, distances in wasserstein_distances.items():
    plt.plot(cycle_ranges, distances, marker='o', linestyle='-', label=col)

# 绘制平均线
plt.plot(cycle_ranges, average_distances, marker='o', linestyle='--', color='black', label='Average')
plt.ylim()
plt.title('1-Wasserstein Distance between Features')
plt.xlabel('Cycles')
plt.ylabel('1-Wasserstein Distance')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 读取Excel文件
file_path_1 = r'E:\MIT\Filtered_Features_MIT.xlsx'
file_path_2 = r'E:\MIT\Filtered_Features_MIT_14.xlsx'

df1 = pd.read_excel(file_path_1).iloc[:700, :]
df2 = pd.read_excel(file_path_2).iloc[:700, :]

# 定义需要计算1-Wasserstein距离的列，排除Q1和Q2
common_columns = list(set(df1.columns) & set(df2.columns))
#columns_to_analyze = [col for col in common_columns if col in ['Vg2', 'RL2', 'VC21', 'VD2', 'ReVC', 'ReVD']]
#columns_to_analyze = [col for col in common_columns if col in [ 'Vg1','Vg2', 'RL1', 'RL2', 'RO', 'VC21', 'VD2', 'tVD2','ReVC', 'ReVD', 'tReVD']]
columns_to_analyze=[ 'Vg1','Vg2', 'RL1', 'RL2', 'RO', 'VC21', 'VD2','ReVC', 'ReVD']
columns_to_analyze = [ 'Vg1','Vg2', 'Q1','Q2','RL1', 'RL2', 'RO', 'VC21', 'VD2','tVD2','ReVC', 'ReVD','tReVD']
#columns_to_analyze=[ 'Vg2',  'RL2',  'ReVC', 'ReVD']
# 归一化处理
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

df1[columns_to_analyze] = scaler1.fit_transform(df1[columns_to_analyze])
df2[columns_to_analyze] = scaler2.fit_transform(df2[columns_to_analyze])

# 计算1-Wasserstein距离函数
def calculate_wasserstein_distance(df1, df2, cycles, columns):
    distances = {}
    for col in columns:
        data1 = df1.iloc[:cycles][col].dropna()
        data2 = df2.iloc[:cycles][col].dropna()
        if not data1.empty and not data2.empty:
            distance = wasserstein_distance(data1, data2)
            distances[col] = 1 - distance
    return distances

# 定义周期范围
max_cycles = min(df1['Cycle_Index'].max(), df2['Cycle_Index'].max())
cycle_ranges = range(20, max_cycles + 1, 20)

# 计算不同周期范围下每个特征的1-Wasserstein距离
wasserstein_distances = {col: [] for col in columns_to_analyze}
average_distances = []

for cycle_range in cycle_ranges:
    distances = calculate_wasserstein_distance(df1, df2, cycle_range, columns_to_analyze)
    for col, distance in distances.items():
        wasserstein_distances[col].append(distance)
    # 计算当前周期范围内所有特征的平均值
    if distances:
        average_distance = np.mean(list(distances.values()))
        average_distances.append(average_distance)
    else:
        average_distances.append(0)

# 绘制1-Wasserstein距离的图表
plt.figure(figsize=(10, 6),dpi=600)

for col, distances in wasserstein_distances.items():
    plt.plot(cycle_ranges, distances, marker='o', linestyle='-', label=col)

# 绘制平均线
plt.plot(cycle_ranges, average_distances, marker='o', linestyle='--', color='black', label='Average')
plt.ylim()
plt.title('1-Wasserstein Distance between Features')
plt.xlabel('Cycles')
plt.ylabel('1-Wasserstein Distance')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# MIT PC TC excel
import pandas as pd
import numpy as np

df_wasserstein = pd.DataFrame(wasserstein_distances)
correlations_df = pd.DataFrame(correlations)
correlations_df_6C = pd.DataFrame(correlations_6C)

correlations_df = correlations_df.T
df_correlation = pd.DataFrame(correlations_df)
correlations_df_6C = correlations_df_6C.T
df_correlation_6C = pd.DataFrame(correlations_df_6C)

file_path = 'D:\\桌面上的文件\\MIT_allF.xlsx'
with pd.ExcelWriter(file_path) as writer:
    df_wasserstein.to_excel(writer, sheet_name='TC_Trend', index=False)
    df_correlation.to_excel(writer, sheet_name='PC_Trend', index=False)
    df_correlation_6C.to_excel(writer, sheet_name='PC_Trend_6C', index=False)

print(f"数据已成功写入 {file_path}")


# In[ ]:


#PC TC Trend
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'D:\桌面上的文件\MIT_allF.xlsx'  
selected_columns = ['Vg2', 'RL2',  'ReVC', 'ReVD','VC21','tVD2','RO']
df_TC = pd.read_excel(file_path, sheet_name='TC_Trend',usecols=selected_columns)  
df_PC = pd.read_excel(file_path, sheet_name='PC_Trend',usecols=selected_columns)  

mean_wasserstein = df_TC.mean(axis=1)
mean_correlation = df_PC.mean(axis=1)

fig, ax1 = plt.subplots(figsize=(4, 2), dpi=600)

color = '#599CB4'  
ax1.set_xlabel('Periods')
ax1.set_ylabel('Average Wasserstein Trend', color=color)
ax1.plot(mean_wasserstein[:21], marker='o', markersize=10,linestyle='-', linewidth=4,color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0.8, 1.05)  

ax2 = ax1.twinx()  
color = '#C25759'  
ax2.set_ylabel('Average Correlation Trend', color=color)
ax2.plot(mean_correlation[:21], marker='o', markersize=10,linestyle='-', linewidth=4,color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0.4, 1.05)  
plt.show()


# In[51]:


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

file_path_5 = r'D:\桌面上的文件\Data Sufficiency\MIT\Filtered_Features_MIT_14.xlsx'
#selected_columns = ['Vg2', 'RL2', 'VC21', 'VD2', 'ReVC', 'ReVD']
selected_columns = ['Vg2', 'RL2',  'ReVC', 'ReVD','RO','VC21','tVD2']
#selected_columns = ['Vg2', 'RL2', 'ReVC', 'ReVD']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[:750]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_5 = scaler.fit_transform(Feature_5)

scaledFeature_5 = pd.DataFrame(data=scaledFeature_5)

print(scaledFeature_5.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_5 = time_series_to_supervised(scaledFeature_5,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
cap=(pd.read_excel(file_path_5).iloc[:750,-1]).values.reshape(-1, 1)
scaledCapacity_5 = scaler.fit_transform(cap)

n_steps_in =3 
n_steps_out=1
processedCapacity_5 = time_series_to_supervised(scaledCapacity_5,n_steps_in,n_steps_out)

data_x5 = processedFeature_5.loc[:,'0(t-3)':'6(t-1)']
data_y5=processedCapacity_5.loc[:,'0']

train_X5=data_x5.values[:750]
train_y5=data_y5[:750]
train_X5 = train_X5.reshape((train_X5.shape[0], n_steps_in, 7))


# In[52]:


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

history = model.fit(train_X5, train_y5, epochs=30, batch_size=64,verbose=2, shuffle=False)#epoch训练的轮数，batch_size每个批次的样本数量

end_time = time.time()
elapsed_time = end_time - start_time
#print(f"模型运行时间：{elapsed_time} 秒")


# In[ ]:


# Automatically observe the situation of different fine-tuning periods
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

file_path_5 = r'E:\MIT\Filtered_Features_MIT_14.xlsx'  
#selected_columns = ['Vg2', 'RL2', 'VC21', 'VD2', 'ReVC', 'ReVD']
selected_columns = ['Vg2', 'RL2','ReVC', 'ReVD','RO','VC21','tVD2']
selected_columns = ['Vg2', 'RL2','ReVC', 'ReVD']
#selected_columns =[ 'Vg1','Vg2', 'RL1', 'RL2', 'RO', 'VC21', 'VD2','ReVC', 'ReVD']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[:800]
file_path_6 = r'E:\MIT\Filtered_Features_MIT.xlsx'  
Feature_6 = pd.read_excel(file_path_6, usecols=selected_columns).iloc[:1020]

capacity_5=pd.read_excel(file_path_5).iloc[:800,-1]
capacity_6=pd.read_excel(file_path_6).iloc[:1020,-1]
cap_6=(pd.read_excel(file_path_6).iloc[:1020,-1]).values.reshape(-1, 1)
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

    data_x6 = processedFeature_6.loc[:,'0(t-3)':'3(t-1)']
    data_y6=processedCapacity_6.loc[:,'0']
    data_y6=data_y6.values.reshape(-1,1)
    
    train_X6=data_x6.values[:train_samples]
    test_X6=data_x6.values[train_samples:]
    train_y6=data_y6[:train_samples]
    test_y6=data_y6[train_samples:]
    train_X6 = train_X6.reshape((train_X6.shape[0], n_steps_in, 4))
    test_X6 = test_X6.reshape((test_X6.shape[0], n_steps_in, 4))
  
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


# MIT Acc
import pandas as pd

prediction_df = pd.DataFrame(prediction_capability, columns=["6C to 4C"])

file_path_output = "D:\桌面上的文件\MIT_Acc.xlsx"

prediction_df.to_excel(file_path_output, index=False)


# In[9]:


# MIT Acc plot
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'D:\桌面上的文件\Data Sufficiency\MIT\MIT_Acc.xlsx'  # 将 'your_file.xlsx' 替换为实 际的文件路径
df = pd.read_excel(file_path)

column_data = df['6C to 4C']
plt.figure(figsize=(5, 2), dpi=600)

for i in range(len(column_data)):
    if i == 3:  # 第四个柱子
        plt.bar(i, column_data[i], width=0.8, color='#898988')
    else:
        plt.bar(i, column_data[i], width=0.8, color='#d7d7d7')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.grid(False)
plt.ylim(0.991,1)
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

from sklearn.decomposition import PCA
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

file_path_5 = r'D:\桌面上的文件\Data Sufficiency\MIT\Filtered_Features_MIT_14.xlsx'
#selected_columns = ['Vg2', 'RL2', 'VC21', 'VD2', 'ReVC', 'ReVD']
selected_columns = ['Vg2', 'RL2',  'ReVC', 'ReVD','RO','VC21','tVD2']
#selected_columns = ['Vg2', 'RL2', 'ReVC', 'ReVD']
Feature_5 = pd.read_excel(file_path_5, usecols=selected_columns).iloc[:750]

scaler = MinMaxScaler(feature_range=(0,1))
scaledFeature_5 = scaler.fit_transform(Feature_5)

scaledFeature_5 = pd.DataFrame(data=scaledFeature_5)

print(scaledFeature_5.shape)

n_steps_in =3 
n_steps_out=1
processedFeature_5 = time_series_to_supervised(scaledFeature_5,n_steps_in,n_steps_out)

scaler = MinMaxScaler(feature_range=(0, 1))
cap=(pd.read_excel(file_path_5).iloc[:750,-1]).values.reshape(-1, 1)
scaledCapacity_5 = scaler.fit_transform(cap)

n_steps_in =3 
n_steps_out=1
processedCapacity_5 = time_series_to_supervised(scaledCapacity_5,n_steps_in,n_steps_out)

data_x5 = processedFeature_5.loc[:,'0(t-3)':'6(t-1)']
data_y5=processedCapacity_5.loc[:,'0']

train_X5=data_x5.values[:750]
train_y5=data_y5[:750]
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

history = model.fit(train_X5, train_y5, epochs=30, batch_size=64,verbose=2, shuffle=False)#epoch训练的轮数，batch_size每个批次的样本数量

end_time = time.time()
elapsed_time = end_time - start_time
#print(f"模型运行时间：{elapsed_time} 秒")


# In[58]:


# 14 Validation no finetune
import matplotlib
train_samples=80

selected_columns = ['Vg2', 'RL2','ReVC', 'ReVD']
selected_columns = ['Vg2', 'RL2',  'ReVC', 'ReVD','RO','VC21','tVD2']
file_path_6 = r'D:\桌面上的文件\Data Sufficiency\MIT\Filtered_Features_MIT.xlsx'  
Feature_6 = pd.read_excel(file_path_6,usecols=selected_columns).iloc[:950]
capacity_6=pd.read_excel(file_path_6).iloc[:950,-1]
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
plt.plot(pd.read_excel(file_path_6).iloc[:950,-1], label='True')
x_range = range(train_samples, train_samples+ len(inv_forecast_y6))
plt.plot(x_range,inv_forecast_y6,marker='.',label='LSTM',linestyle=None,markersize=5)

plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[59]:


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
plt.plot(pd.read_excel(file_path_6).iloc[:950,-1], label='True')
x_range = range(train_samples, train_samples+len(inv_forecast_y6t))
plt.plot(x_range,inv_forecast_y6t,marker='.',label='LSTM+Fine-tune',linestyle=None,markersize=5)
plt.ylabel('Capacity(Ah)',fontsize=12)
plt.xlabel('Cycle',fontsize=12)
plt.axvline(x=train_samples, color='gray', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[77]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel(file_path_6)
initial_capacity = data.iloc[0, -1]  
threshold_capacity = 0.95 * initial_capacity  

plt.figure(figsize=(4, 3),dpi=600)
plt.plot(data.iloc[:950, -1], label='Target_True',linewidth=5,color=plt.cm.Blues(0.8))
plt.plot(pd.read_excel(file_path_5).iloc[:750,-1], label='Source',linewidth=5,color=plt.cm.Reds(0.8))

x_range = range(train_samples, train_samples + len(inv_forecast_y6t))
plt.plot(x_range, inv_forecast_y6t,label='Target_Pre', linestyle=None,linewidth=5,color=plt.cm.Greens(0.8))

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
plt.title('MIT-6C')
plt.ylim(initial_capacity*0.7,initial_capacity*1.05)
plt.show()

print(f"真实容量曲线与 0.8 初始容量的交点对应的循环数: {true_x_intersection}")
print(f"预测容量曲线与 0.8 初始容量的交点对应的循环数: {pred_x_intersection}")


# In[ ]:




