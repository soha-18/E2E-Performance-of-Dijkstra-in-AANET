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
    
#function to calculate confidence interval
def confidence_interval_t(data, confidence=0.95):
    data_array = 1.0 * np.array(data)
    degree_of_freedom = len(data_array) - 1
    sample_mean, sample_standard_error = np.mean(data_array), stats.sem(data_array)
    t = stats.t.ppf((1 + confidence) / 2., degree_of_freedom)
    margin_of_error = sample_standard_error * t
    Confidence_Interval = 1.0 * np.array([sample_mean - margin_of_error, sample_mean + margin_of_error])
    return sample_mean, Confidence_Interval, margin_of_error

# file_path = 'E:/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/DSPR/queueLength_vector.csv'
# file_path_gpsr = 'E:/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/GPSR/queueLength_vector.csv'
# file_path = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/NAC_Scenario/DSPR/queueLength_vector.csv'
# file_path_gpsr = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/NAC_Scenario/GPSR/queueLength_vector.csv'
file_path = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/queueLength_vector.csv'
file_path_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/queueLength_vector.csv'

queueLength_vector = pd.read_csv(file_path)
queueLength_vector_gpsr = pd.read_csv(file_path_gpsr)

column_to_check = 'name'
queueLength_vector = queueLength_vector.dropna(subset=[column_to_check])
queueLength_vector_gpsr = queueLength_vector_gpsr.dropna(subset=[column_to_check])


queueLength_vector = process_dataframe(queueLength_vector)
queueLength_vector_gpsr = process_dataframe(queueLength_vector_gpsr)

queueLength_vector['vecvalue'] = queueLength_vector['vecvalue'].str.split()
queueLength_vector_gpsr['vecvalue'] = queueLength_vector_gpsr['vecvalue'].str.split()

queueLength_vector = pd.DataFrame({
    'sendInterval': queueLength_vector['sendInterval'].repeat(queueLength_vector['vecvalue'].apply(len)),
    'SimulationRun': queueLength_vector['SimulationRun'].repeat(queueLength_vector['vecvalue'].apply(len)),
    'vecvalue': [x for sublist in queueLength_vector['vecvalue'] for x in sublist]
})
queueLength_vector['vecvalue'] = pd.to_numeric(queueLength_vector['vecvalue'], errors='coerce')

queueLength_vector_gpsr = pd.DataFrame({
    'sendInterval': queueLength_vector_gpsr['sendInterval'].repeat(queueLength_vector_gpsr['vecvalue'].apply(len)),
    'SimulationRun': queueLength_vector_gpsr['SimulationRun'].repeat(queueLength_vector_gpsr['vecvalue'].apply(len)),
    'vecvalue': [x for sublist in queueLength_vector_gpsr['vecvalue'] for x in sublist]
})
queueLength_vector_gpsr['vecvalue'] = pd.to_numeric(queueLength_vector_gpsr['vecvalue'], errors='coerce')

sendInterval_values = queueLength_vector['sendInterval'].unique()
# print(sendInterval_values)
sent_packets_intervals = ['90s', '110s', '130s', '150s', '170s']
# sent_packets_intervals = ['1ms', '10ms', '100ms', '1000ms', '10000ms']

#print(np.mean(queueLength_vector['vecvalue'].to_numpy()))
#print(np.mean(queueLength_vector_gpsr['vecvalue'].to_numpy()))
#print(queueLength_vector)

#filter_df = queueLength_vector[queueLength_vector['sendInterval'] == 90]
#filter_df1 = queueLength_vector_gpsr[queueLength_vector_gpsr['sendInterval'] == 150]
#print(np.mean(filter_df['vecvalue'].to_numpy()))
#print(np.mean(filter_df1['vecvalue'].to_numpy()))


sample_mean_Q_length = []
sample_mean_Q_length_gpsr = []
margin_of_error_array = []
margin_of_error_array_gpsr = []

for timestamp in sendInterval_values:
    group1 = queueLength_vector[queueLength_vector['sendInterval'] == timestamp]
    group2 = queueLength_vector_gpsr[queueLength_vector_gpsr['sendInterval'] == timestamp]
    queue_length = group1['vecvalue']
    queue_length_gpsr = group2['vecvalue']
    queue_length = queue_length.dropna()
    queue_length_gpsr = queue_length_gpsr.dropna()
    sample_mean1, confidence_Interval1, margin_of_error1 = confidence_interval_t(queue_length, confidence=0.95)
    sample_mean2, confidence_Interval2, margin_of_error2 = confidence_interval_t(queue_length_gpsr, confidence=0.95)
    sample_mean_Q_length =  np.append(sample_mean_Q_length, sample_mean1)
    sample_mean_Q_length_gpsr =  np.append(sample_mean_Q_length_gpsr, sample_mean2)
    margin_of_error_array = np.append(margin_of_error_array, margin_of_error1)
    margin_of_error_array_gpsr = np.append(margin_of_error_array_gpsr, margin_of_error2)


plt.figure(figsize=(10, 6))
plt.errorbar(x=sent_packets_intervals,y= sample_mean_Q_length,yerr=margin_of_error_array,fmt='D',capsize=4, capthick=1, color='m', markersize=8,label='DSPR') #ecolor='c'
plt.errorbar(x=sent_packets_intervals,y= sample_mean_Q_length_gpsr,yerr=margin_of_error_array_gpsr,fmt='o',capsize=4, capthick=1, color='b', markersize=8,label='GPSR')  #ecolor='r'
plt.xlabel('Packet sending Intervals')
plt.ylabel('Queue Length')
plt.tight_layout()
plt.title('Average Queue Length using Confidence Intervals')
plt.legend()
plt.grid(True)
# plt.savefig('queueLength_12_nodes_scenario_CI.pdf', format='pdf', dpi=1200)
#plt.savefig('queueLength_NAC_A2G_scenario_CI.pdf', format='pdf', dpi=1200)
plt.show()