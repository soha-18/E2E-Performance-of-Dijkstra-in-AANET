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
   
#function to calculate confidence interval
def confidence_interval_t(data, confidence=0.95):
    data_array = 1.0 * np.array(data)
    degree_of_freedom = len(data_array) - 1
    sample_mean, sample_standard_error = np.mean(data_array), stats.sem(data_array)
    t = stats.t.ppf((1 + confidence) / 2., degree_of_freedom)
    margin_of_error = sample_standard_error * t
    Confidence_Interval = 1.0 * np.array([sample_mean - margin_of_error, sample_mean + margin_of_error])
    return sample_mean, Confidence_Interval, margin_of_error


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
# sendInterval_values1 = e2e_delay_vector1['sendInterval'].unique()
# print(sendInterval_values)
sent_packets_intervals = ['50', '70', '90', '110', '130', '150', '170']
# sent_packets_intervals = ['1ms', '10ms', '100ms', '1000ms', '10000ms']

sample_mean_E2E_delay = []
margin_of_error_array = []
sample_mean_E2E_delay_gpsr = []
margin_of_error_array_gpsr = []

for timestamp in sendInterval_values:
    group1 = e2e_delay_vector2[e2e_delay_vector2['sendInterval'] == timestamp]
    group2 = e2e_delay_vector_gpsr2[e2e_delay_vector_gpsr2['sendInterval'] == timestamp]
    e2e_delay = group1['vecvalue']
    e2e_delay_gpsr = group2['vecvalue']
    e2e_delay = e2e_delay.dropna()
    e2e_delay_gpsr = e2e_delay_gpsr.dropna()
    sample_mean1, confidence_Interval1, margin_of_error1 = confidence_interval_t(e2e_delay, confidence=0.95)
    sample_mean2, confidence_Interval2, margin_of_error2 = confidence_interval_t(e2e_delay_gpsr, confidence=0.95)
    sample_mean_E2E_delay =  np.append(sample_mean_E2E_delay, sample_mean1)
    sample_mean_E2E_delay_gpsr =  np.append(sample_mean_E2E_delay_gpsr, sample_mean2)
    margin_of_error_array = np.append(margin_of_error_array, margin_of_error1)
    margin_of_error_array_gpsr = np.append(margin_of_error_array_gpsr, margin_of_error2)


# # # CI Plot
plt.errorbar(x = sent_packets_intervals, y= sample_mean_E2E_delay, yerr=margin_of_error_array, fmt='D',capsize=4, capthick=1, color='m', markersize=8,label='DSPR')
plt.errorbar(x = sent_packets_intervals, y= sample_mean_E2E_delay_gpsr, yerr=margin_of_error_array_gpsr, fmt='o',capsize=4, capthick=1, color='b', markersize=8,label='GPSR')
plt.xlabel('Packet sending Intervals in seconds')
plt.ylabel('E2E Delay in seconds')
plt.legend()
plt.grid(True)
# plt.savefig('E2E Delay_12_nodes_scenario_CI.pdf', format='pdf', dpi=1200)
# plt.savefig('E2E Delay_NAC_scenario_CI.pdf', format='pdf', dpi=1200)
# plt.savefig('E2E Delay_NAC_A2G_scenario_CI.pdf', format='pdf', dpi=1200)
plt.show()

