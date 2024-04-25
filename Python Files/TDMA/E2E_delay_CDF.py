import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats



def process_dataframe(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
    df.reset_index(drop=True, inplace=True)
    df['sendInterval'] = df['run'].apply(calculate_send_interval)
    df = df[['sendInterval'] + [col for col in df.columns if col != 'sendInterval']]
    # df = df.drop(columns=['vectime'])
    df = df.rename(columns={'run': 'SimulationRun'})
    return df

def process_dataframe1(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
    df.reset_index(drop=True, inplace=True)
    df['sendInterval'] = df['run'].apply(calculate_send_interval1)
    df = df[['sendInterval'] + [col for col in df.columns if col != 'sendInterval']]
    # df = df.drop(columns=['vectime'])
    df = df.rename(columns={'run': 'SimulationRun'})
    return df

def calculate_send_interval(run_value):
    run_value = int(run_value)
    if 0 <= run_value <= 9:
        return 90
        # return 1
    elif 10 <= run_value <= 19:
        return 110
        # return 10
    elif 20 <= run_value <= 29:
        return 130
        # return 100
    elif 30 <= run_value <= 39:
        return 150
        # return 1000
    else:
        return 170
        # return 10000

def calculate_send_interval1(run_value):
    run_value = int(run_value)
    if 0 <= run_value <= 9:
        return 50
    else:
        return 70

#function to calculate cdf of a dataset
def calculate_cdf(dataset):
    cdf_data = []
    sorted_data = []
    for i in range(0,len(dataset)):
        column = dataset[i]
        sorted_column = sorted(column)
        sorted_data.append(sorted_column)
        cdf = np.arange(1, len(sorted_column) + 1) / len(sorted_column)
        cdf_data.append(cdf)
    return sorted_data, cdf_data

# file_path = 'E:/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/DSPR/e2e_delay_vector.csv'
# file_path = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/NAC_Scenario/DSPR/e2e_delay_vector.csv'
file_path = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/e2e_delay_vector.csv'
file_path1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/e2e_delay_vector1.csv'
# file_path_gpsr = 'E:/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/GPSR/e2e_delay_vector.csv'
# file_path_gpsr = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/NAC_Scenario/GPSR/e2e_delay_vector.csv'
file_path_gpsr =  '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/e2e_delay_vector.csv'
file_path_gpsr1 =  '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/e2e_delay_vector1.csv'

e2e_delay_vector = pd.read_csv(file_path)
e2e_delay_vector_gpsr = pd.read_csv(file_path_gpsr)

e2e_delay_vector1 = pd.read_csv(file_path1)
e2e_delay_vector_gpsr1 = pd.read_csv(file_path_gpsr1)

column_to_check = 'name'
e2e_delay_vector = e2e_delay_vector.dropna(subset=[column_to_check])
e2e_delay_vector_gpsr = e2e_delay_vector_gpsr.dropna(subset=[column_to_check])
e2e_delay_vector1 = e2e_delay_vector1.dropna(subset=[column_to_check])
e2e_delay_vector_gpsr1 = e2e_delay_vector_gpsr1.dropna(subset=[column_to_check])


e2e_delay_vector = process_dataframe(e2e_delay_vector)
e2e_delay_vector_gpsr = process_dataframe(e2e_delay_vector_gpsr)
e2e_delay_vector['vecvalue'] = e2e_delay_vector['vecvalue'].str.split()
e2e_delay_vector['vectime'] = e2e_delay_vector['vectime'].str.split()
e2e_delay_vector_gpsr['vecvalue'] = e2e_delay_vector_gpsr['vecvalue'].str.split()
e2e_delay_vector_gpsr['vectime'] = e2e_delay_vector_gpsr['vectime'].str.split()

e2e_delay_vector1 = process_dataframe1(e2e_delay_vector1)
e2e_delay_vector_gpsr1 = process_dataframe1(e2e_delay_vector_gpsr1)
e2e_delay_vector1['vecvalue'] = e2e_delay_vector1['vecvalue'].str.split()
e2e_delay_vector1['vectime'] = e2e_delay_vector1['vectime'].str.split()
e2e_delay_vector_gpsr1['vecvalue'] = e2e_delay_vector_gpsr1['vecvalue'].str.split()
e2e_delay_vector_gpsr1['vectime'] = e2e_delay_vector_gpsr1['vectime'].str.split()

e2e_delay_vector = pd.DataFrame({
    'sendInterval': e2e_delay_vector['sendInterval'].repeat(e2e_delay_vector['vecvalue'].apply(len)),
    'SimulationRun': e2e_delay_vector['SimulationRun'].repeat(e2e_delay_vector['vecvalue'].apply(len)),
    'vectime': [x for sublist in e2e_delay_vector['vectime'] for x in sublist],
    'vecvalue': [x for sublist in e2e_delay_vector['vecvalue'] for x in sublist]
})
e2e_delay_vector['vecvalue'] = pd.to_numeric(e2e_delay_vector['vecvalue'], errors='coerce')
e2e_delay_vector['vectime'] = pd.to_numeric(e2e_delay_vector['vectime'], errors='coerce')
filtered_e2e_delay_vector = e2e_delay_vector[(e2e_delay_vector['vectime'] >= 200.0) & (e2e_delay_vector['vectime'] <= 1500.0)]
filtered_e2e_delay_vector = filtered_e2e_delay_vector.drop(columns=['vectime'])
filtered_e2e_delay_vector.reset_index(drop=True, inplace=True)

e2e_delay_vector_gpsr = pd.DataFrame({
    'sendInterval': e2e_delay_vector_gpsr['sendInterval'].repeat(e2e_delay_vector_gpsr['vecvalue'].apply(len)),
    'SimulationRun': e2e_delay_vector_gpsr['SimulationRun'].repeat(e2e_delay_vector_gpsr['vecvalue'].apply(len)),
    'vectime': [x for sublist in e2e_delay_vector_gpsr['vectime'] for x in sublist],
    'vecvalue': [x for sublist in e2e_delay_vector_gpsr['vecvalue'] for x in sublist]
})
e2e_delay_vector_gpsr['vecvalue'] = pd.to_numeric(e2e_delay_vector_gpsr['vecvalue'], errors='coerce')
e2e_delay_vector_gpsr['vectime'] = pd.to_numeric(e2e_delay_vector_gpsr['vectime'], errors='coerce')
filtered_e2e_delay_vector_gpsr = e2e_delay_vector_gpsr[(e2e_delay_vector_gpsr['vectime'] >= 200.0) & (e2e_delay_vector_gpsr['vectime'] <= 1500.0)]
filtered_e2e_delay_vector_gpsr = filtered_e2e_delay_vector_gpsr.drop(columns=['vectime'])
filtered_e2e_delay_vector_gpsr.reset_index(drop=True, inplace=True)

e2e_delay_vector1 = pd.DataFrame({
    'sendInterval': e2e_delay_vector1['sendInterval'].repeat(e2e_delay_vector1['vecvalue'].apply(len)),
    'SimulationRun': e2e_delay_vector1['SimulationRun'].repeat(e2e_delay_vector1['vecvalue'].apply(len)),
    'vectime': [x for sublist in e2e_delay_vector1['vectime'] for x in sublist],
    'vecvalue': [x for sublist in e2e_delay_vector1['vecvalue'] for x in sublist]
})
e2e_delay_vector1['vecvalue'] = pd.to_numeric(e2e_delay_vector1['vecvalue'], errors='coerce')
e2e_delay_vector1['vectime'] = pd.to_numeric(e2e_delay_vector1['vectime'], errors='coerce')
filtered_e2e_delay_vector1 = e2e_delay_vector1[(e2e_delay_vector1['vectime'] >= 200.0) & (e2e_delay_vector1['vectime'] <= 1500.0)]
filtered_e2e_delay_vector1 = filtered_e2e_delay_vector1.drop(columns=['vectime'])
filtered_e2e_delay_vector1.reset_index(drop=True, inplace=True)

e2e_delay_vector_gpsr1 = pd.DataFrame({
    'sendInterval': e2e_delay_vector_gpsr1['sendInterval'].repeat(e2e_delay_vector_gpsr1['vecvalue'].apply(len)),
    'SimulationRun': e2e_delay_vector_gpsr1['SimulationRun'].repeat(e2e_delay_vector_gpsr1['vecvalue'].apply(len)),
    'vectime': [x for sublist in e2e_delay_vector_gpsr1['vectime'] for x in sublist],
    'vecvalue': [x for sublist in e2e_delay_vector_gpsr1['vecvalue'] for x in sublist]
})
e2e_delay_vector_gpsr1['vecvalue'] = pd.to_numeric(e2e_delay_vector_gpsr1['vecvalue'], errors='coerce')
e2e_delay_vector_gpsr1['vectime'] = pd.to_numeric(e2e_delay_vector_gpsr1['vectime'], errors='coerce')
filtered_e2e_delay_vector_gpsr1 = e2e_delay_vector_gpsr1[(e2e_delay_vector_gpsr1['vectime'] >= 200.0) & (e2e_delay_vector_gpsr1['vectime'] <= 1500.0)]
filtered_e2e_delay_vector_gpsr1 = filtered_e2e_delay_vector_gpsr1.drop(columns=['vectime'])
filtered_e2e_delay_vector_gpsr1.reset_index(drop=True, inplace=True)

e2e_delay_vector2 = pd.concat([filtered_e2e_delay_vector1, filtered_e2e_delay_vector])
e2e_delay_vector_gpsr2 = pd.concat([filtered_e2e_delay_vector_gpsr1, filtered_e2e_delay_vector_gpsr])

# print(e2e_delay_vector_gpsr2)

sendInterval_values = e2e_delay_vector_gpsr2['sendInterval'].unique()
sendInterval_values = sorted(sendInterval_values)


for i, timestamp in enumerate(sendInterval_values):
    group = e2e_delay_vector2[e2e_delay_vector2['sendInterval'] == timestamp]
    group1 = e2e_delay_vector_gpsr2[e2e_delay_vector_gpsr2['sendInterval'] == timestamp]
    e2e_delay = group['vecvalue']
    e2e_delay_gpsr = group1['vecvalue']
    sorted_data = np.sort(e2e_delay)
    sorted_data_gpsr = np.sort(e2e_delay_gpsr)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    cdf_gpsr = np.arange(1, len(sorted_data_gpsr) + 1) / len(sorted_data_gpsr)
    linestyle = '--' if i % 2 == 0 else '-'
    plt.step(sorted_data, cdf, label=f'{timestamp}s',  linestyle=linestyle)
    # plt.step(sorted_data_gpsr, cdf_gpsr, label=f'{timestamp}s',  linestyle=linestyle)


# plt.title('E2E Delay Distribution ')
plt.xlabel('E2E Delay in seconds')
plt.ylabel('CDF')
plt.legend()
plt.grid(True)
# plt.savefig('E2E_Delay_NAC_A2G_scenario_CDF_DSPR.pdf', format='pdf', dpi=1200)
# plt.savefig('E2E_Delay_NAC_A2G_scenario_CDF_GPSR.pdf', format='pdf', dpi=1200)
plt.show()
