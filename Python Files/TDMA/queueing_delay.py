import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def process_dataframe(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
    df.reset_index(drop=True, inplace=True)
    df['sendInterval'] = df['run'].apply(calculate_send_interval)
    df = df[['sendInterval'] + [col for col in df.columns if col != 'sendInterval']]
    df = df.drop(columns=['vectime'])
    df = df.rename(columns={'run': 'SimulationRun'})
    return df

def process_dataframe1(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
    df.reset_index(drop=True, inplace=True)
    df['sendInterval'] = df['run'].apply(calculate_send_interval1)
    df = df[['sendInterval'] + [col for col in df.columns if col != 'sendInterval']]
    df = df.drop(columns=['vectime'])
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

# file_path = 'E:/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/DSPR/queueingTime_vector.csv'
# file_path_gpsr = 'E:/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/GPSR/queueingTime_vector.csv'
# file_path = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/NAC_Scenario/DSPR/queueingTime_vector.csv'
# file_path_gpsr = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/NAC_Scenario/GPSR/queueingTime_vector.csv'
file_path = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/queueingTime_vector.csv'
file_path_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/queueingTime_vector.csv'

file_path1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/queueingTime_vector1.csv'
file_path_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/queueingTime_vector1.csv'

queue_delay_vector = pd.read_csv(file_path)
queue_delay_vector_gpsr = pd.read_csv(file_path_gpsr)

queue_delay_vector1 = pd.read_csv(file_path1)
queue_delay_vector_gpsr1 = pd.read_csv(file_path_gpsr1)

column_to_check = 'name'
queue_delay_vector = queue_delay_vector.dropna(subset=[column_to_check])
queue_delay_vector_gpsr = queue_delay_vector_gpsr.dropna(subset=[column_to_check])
queue_delay_vector1 = queue_delay_vector1.dropna(subset=[column_to_check])
queue_delay_vector_gpsr1 = queue_delay_vector_gpsr1.dropna(subset=[column_to_check])


queue_delay_vector = process_dataframe(queue_delay_vector)
queue_delay_vector_gpsr = process_dataframe(queue_delay_vector_gpsr)

queue_delay_vector1 = process_dataframe1(queue_delay_vector1)
queue_delay_vector_gpsr1 = process_dataframe1(queue_delay_vector_gpsr1)

queue_delay_vector['vecvalue'] = queue_delay_vector['vecvalue'].str.split()
queue_delay_vector_gpsr['vecvalue'] = queue_delay_vector_gpsr['vecvalue'].str.split()

queue_delay_vector1['vecvalue'] = queue_delay_vector1['vecvalue'].str.split()
queue_delay_vector_gpsr1['vecvalue'] = queue_delay_vector_gpsr1['vecvalue'].str.split()

queue_delay_vector = pd.DataFrame({
    'sendInterval': queue_delay_vector['sendInterval'].repeat(queue_delay_vector['vecvalue'].apply(len)),
    'SimulationRun': queue_delay_vector['SimulationRun'].repeat(queue_delay_vector['vecvalue'].apply(len)),
    'vecvalue': [x for sublist in queue_delay_vector['vecvalue'] for x in sublist]
})
queue_delay_vector['vecvalue'] = pd.to_numeric(queue_delay_vector['vecvalue'], errors='coerce')

queue_delay_vector1 = pd.DataFrame({
    'sendInterval': queue_delay_vector1['sendInterval'].repeat(queue_delay_vector1['vecvalue'].apply(len)),
    'SimulationRun': queue_delay_vector1['SimulationRun'].repeat(queue_delay_vector1['vecvalue'].apply(len)),
    'vecvalue': [x for sublist in queue_delay_vector1['vecvalue'] for x in sublist]
})
queue_delay_vector1['vecvalue'] = pd.to_numeric(queue_delay_vector1['vecvalue'], errors='coerce')

queue_delay_vector_gpsr = pd.DataFrame({
    'sendInterval': queue_delay_vector_gpsr['sendInterval'].repeat(queue_delay_vector_gpsr['vecvalue'].apply(len)),
    'SimulationRun': queue_delay_vector_gpsr['SimulationRun'].repeat(queue_delay_vector_gpsr['vecvalue'].apply(len)),
    'vecvalue': [x for sublist in queue_delay_vector_gpsr['vecvalue'] for x in sublist]
})
queue_delay_vector_gpsr['vecvalue'] = pd.to_numeric(queue_delay_vector_gpsr['vecvalue'], errors='coerce')

queue_delay_vector_gpsr1 = pd.DataFrame({
    'sendInterval': queue_delay_vector_gpsr1['sendInterval'].repeat(queue_delay_vector_gpsr1['vecvalue'].apply(len)),
    'SimulationRun': queue_delay_vector_gpsr1['SimulationRun'].repeat(queue_delay_vector_gpsr1['vecvalue'].apply(len)),
    'vecvalue': [x for sublist in queue_delay_vector_gpsr1['vecvalue'] for x in sublist]
})
queue_delay_vector_gpsr1['vecvalue'] = pd.to_numeric(queue_delay_vector_gpsr1['vecvalue'], errors='coerce')

queue_delay_vector2 = pd.concat([queue_delay_vector1, queue_delay_vector])
queue_delay_vector_gpsr2 = pd.concat([queue_delay_vector_gpsr1, queue_delay_vector_gpsr])

sendInterval_values = queue_delay_vector2['sendInterval'].unique()
sendInterval_values = sorted(sendInterval_values)
# sendInterval_values1 = queue_delay_vector1['sendInterval'].unique()
# print(sendInterval_values)
sent_packets_intervals = ['50','70','90', '110', '130', '150', '170']
# sent_packets_intervals = ['1ms', '10ms', '100ms', '1000ms', '10000ms']

sample_mean_Q_delay = []
sample_mean_Q_delay_gpsr = []
margin_of_error_array = []
margin_of_error_array_gpsr = []

for timestamp in sendInterval_values:
    group1 = queue_delay_vector2[queue_delay_vector2['sendInterval'] == timestamp]
    group2 = queue_delay_vector_gpsr2[queue_delay_vector_gpsr2['sendInterval'] == timestamp]
    queue_delay = group1['vecvalue']
    queue_delay_gpsr = group2['vecvalue']
    queue_delay = queue_delay.dropna()
    queue_delay_gpsr = queue_delay_gpsr.dropna()
    sample_mean1, confidence_Interval1, margin_of_error1 = confidence_interval_t(queue_delay, confidence=0.95)
    sample_mean2, confidence_Interval2, margin_of_error2 = confidence_interval_t(queue_delay_gpsr, confidence=0.95)
    sample_mean_Q_delay =  np.append(sample_mean_Q_delay, sample_mean1)
    sample_mean_Q_delay_gpsr =  np.append(sample_mean_Q_delay_gpsr, sample_mean2)
    margin_of_error_array = np.append(margin_of_error_array, margin_of_error1)
    margin_of_error_array_gpsr = np.append(margin_of_error_array_gpsr, margin_of_error2)

plt.errorbar(x=sent_packets_intervals,y= sample_mean_Q_delay,yerr=margin_of_error_array,fmt='D',capsize=4, capthick=1, color='m', markersize=8,label='DSPR') #ecolor='c'
plt.errorbar(x=sent_packets_intervals,y= sample_mean_Q_delay_gpsr,yerr=margin_of_error_array_gpsr,fmt='o',capsize=4, capthick=1, color='b', markersize=8,label='GPSR')  #ecolor='r'
plt.xlabel('Packet sending Intervals in seconds')
plt.ylabel('Queuing delay in seconds')
plt.legend()
plt.grid(True)
# plt.savefig('Queuing delay_12_nodes_scenario_CI.pdf', format='pdf', dpi=1200)
# plt.savefig('Queuing delay_NAC_scenario_CI.pdf', format='pdf', dpi=1200)
# plt.savefig('Queuing delay_NAC_A2G_scenario_CI.pdf', format='pdf', dpi=1200)
plt.show()
