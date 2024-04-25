import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def process_dataframe(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue'])
    df.reset_index(drop=True, inplace=True)
    df['sendInterval'] = df['run'].apply(calculate_send_interval)
    df = df[['sendInterval'] + [col for col in df.columns if col != 'sendInterval']]
    df = df.rename(columns={'run': 'SimulationRun'})
    return df

def process_dataframe1(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue'])
    df.reset_index(drop=True, inplace=True)
    df['sendInterval'] = df['run'].apply(calculate_send_interval1)
    df = df[['sendInterval'] + [col for col in df.columns if col != 'sendInterval']]
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

# packet_sent_file = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/DSPR/packet_sent.csv'
# packet_sent_file_gpsr = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/GPSR/packet_sent.csv'
# packet_sent_file = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/NAC_Scenario/DSPR/packet_sent.csv'
# packet_sent_file_gpsr = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/NAC_Scenario/GPSR/packet_sent.csv'
packet_sent_file = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/packet_sent.csv'
packet_sent_file_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/packet_sent.csv'

packet_sent_file1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/packet_sent1.csv'
packet_sent_file_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/packet_sent1.csv'

# packet_received_file = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/DSPR/packet_received.csv'
# packet_received_file_gpsr = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/GPSR/packet_received.csv'
# packet_received_file = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/NAC_Scenario/DSPR/packet_received.csv'
# packet_received_file_gpsr = '/media/sohini/HAHAHA/Study materials/Thesis/Thesis_Code/Results/NAC_Scenario/GPSR/packet_received.csv'
packet_received_file = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/packet_received.csv'
packet_received_file_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/packet_received.csv'

packet_received_file1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/packet_received1.csv'
packet_received_file_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/packet_received1.csv'


sent_vector = pd.read_csv(packet_sent_file)
sent_vector_gpsr = pd.read_csv(packet_sent_file_gpsr)
receive_vector = pd.read_csv(packet_received_file)
receive_vector_gpsr = pd.read_csv(packet_received_file_gpsr)

sent_vector1 = pd.read_csv(packet_sent_file1)
sent_vector_gpsr1 = pd.read_csv(packet_sent_file_gpsr1)
receive_vector1 = pd.read_csv(packet_received_file1)
receive_vector_gpsr1 = pd.read_csv(packet_received_file_gpsr1)

column_to_check = 'name'
sent_vector = sent_vector.dropna(subset=[column_to_check])
sent_vector_gpsr = sent_vector_gpsr.dropna(subset=[column_to_check])
sent_vector1 = sent_vector1.dropna(subset=[column_to_check])
sent_vector_gpsr1 = sent_vector_gpsr1.dropna(subset=[column_to_check])

sent_vector = process_dataframe(sent_vector)
sent_vector_gpsr = process_dataframe(sent_vector_gpsr)
sent_vector1 = process_dataframe1(sent_vector1)
sent_vector_gpsr1 = process_dataframe1(sent_vector_gpsr1)

# print(sent_vector)
receive_vector = receive_vector.dropna(subset=[column_to_check])
receive_vector_gpsr = receive_vector_gpsr.dropna(subset=[column_to_check])
receive_vector1 = receive_vector1.dropna(subset=[column_to_check])
receive_vector_gpsr1 = receive_vector_gpsr1.dropna(subset=[column_to_check])

receive_vector = process_dataframe(receive_vector)
receive_vector_gpsr = process_dataframe(receive_vector_gpsr)
receive_vector1 = process_dataframe1(receive_vector1)
receive_vector_gpsr1 = process_dataframe1(receive_vector_gpsr1)

sent_vector = sent_vector.rename(columns={'value': 'Packet_Sent'})
sent_vector = sent_vector.groupby(['sendInterval', 'SimulationRun'])['Packet_Sent'].sum().reset_index()
sent_vector_gpsr = sent_vector_gpsr.rename(columns={'value': 'Packet_Sent'})
sent_vector_gpsr = sent_vector_gpsr.groupby(['sendInterval', 'SimulationRun'])['Packet_Sent'].sum().reset_index()
receive_vector = receive_vector.rename(columns={'value': 'Packet_Received'})
receive_vector_gpsr = receive_vector_gpsr.rename(columns={'value': 'Packet_Received'})

sent_vector1 = sent_vector1.rename(columns={'value': 'Packet_Sent'})
sent_vector1 = sent_vector1.groupby(['sendInterval', 'SimulationRun'])['Packet_Sent'].sum().reset_index()
sent_vector_gpsr1 = sent_vector_gpsr1.rename(columns={'value': 'Packet_Sent'})
sent_vector_gpsr1 = sent_vector_gpsr1.groupby(['sendInterval', 'SimulationRun'])['Packet_Sent'].sum().reset_index()
receive_vector1 = receive_vector1.rename(columns={'value': 'Packet_Received'})
receive_vector_gpsr1 = receive_vector_gpsr1.rename(columns={'value': 'Packet_Received'})


PDR_df = pd.merge(sent_vector, receive_vector, on=['sendInterval', 'SimulationRun'], how='inner')
PDR_df_gpsr = pd.merge(sent_vector_gpsr, receive_vector_gpsr, on=['sendInterval', 'SimulationRun'], how='inner')
PDR_df['PDR'] = PDR_df['Packet_Received']/PDR_df['Packet_Sent']
PDR_df_gpsr['PDR'] = PDR_df_gpsr['Packet_Received']/PDR_df_gpsr['Packet_Sent']
# PDR_df = PDR_df.drop(columns=['Packet_Received', 'Packet_Sent'])
# PDR_df_gpsr = PDR_df_gpsr.drop(columns=['Packet_Received', 'Packet_Sent'])
# print(PDR_df)
# print(PDR_df_gpsr)

PDR_df1 = pd.merge(sent_vector1, receive_vector1, on=['sendInterval', 'SimulationRun'], how='inner')
PDR_df_gpsr1 = pd.merge(sent_vector_gpsr1, receive_vector_gpsr1, on=['sendInterval', 'SimulationRun'], how='inner')
PDR_df1['PDR'] = PDR_df1['Packet_Received']/PDR_df1['Packet_Sent']
PDR_df_gpsr1['PDR'] = PDR_df_gpsr1['Packet_Received']/PDR_df_gpsr1['Packet_Sent']

# print(PDR_df_gpsr1)

sendInterval_values = receive_vector['sendInterval'].unique()
sendInterval_values = sorted(sendInterval_values)
sendInterval_values1 = receive_vector1['sendInterval'].unique()
# print(sendInterval_values)
sent_packets_intervals = ['50','70','90', '110', '130', '150', '170']
# sent_packets_intervals = ['1ms', '10ms', '100ms', '1000ms', '10000ms']

sample_mean_PDR_array = []
margin_of_error_array = []
sample_mean_PDR_array_gpsr = []
margin_of_error_array_gpsr = []

for timestamp in sendInterval_values:
    group1 = PDR_df[PDR_df['sendInterval'] == timestamp]
    group2 = PDR_df_gpsr[PDR_df_gpsr['sendInterval'] == timestamp]
    pdr = group1['PDR']
    pdr_gpsr = group2['PDR']
    pdr = pdr.dropna()
    pdr_gpsr = pdr_gpsr.dropna()
    sample_mean1, confidence_Interval1, margin_of_error1 = confidence_interval_t(pdr, confidence=0.95)
    sample_mean2, confidence_Interval2, margin_of_error2 = confidence_interval_t(pdr_gpsr, confidence=0.95)
    sample_mean_PDR_array =  np.append(sample_mean_PDR_array, sample_mean1)
    sample_mean_PDR_array_gpsr =  np.append(sample_mean_PDR_array_gpsr, sample_mean2)
    margin_of_error_array = np.append(margin_of_error_array, margin_of_error1)
    margin_of_error_array_gpsr = np.append(margin_of_error_array_gpsr, margin_of_error2)

# print(sample_mean_PDR_array)
# print(sample_mean_PDR_array_gpsr)
   
sample_mean_PDR_array1 = []
margin_of_error_array1 = []
sample_mean_PDR_array_gpsr1 = []
margin_of_error_array_gpsr1 = []    
    
for timestamp in sendInterval_values1:
    group1 = PDR_df1[PDR_df1['sendInterval'] == timestamp]
    group2 = PDR_df_gpsr1[PDR_df_gpsr1['sendInterval'] == timestamp]
    pdr = group1['PDR']
    pdr_gpsr = group2['PDR']
    pdr = pdr.dropna()
    pdr_gpsr = pdr_gpsr.dropna()
    sample_mean1, confidence_Interval1, margin_of_error1 = confidence_interval_t(pdr, confidence=0.95)
    sample_mean2, confidence_Interval2, margin_of_error2 = confidence_interval_t(pdr_gpsr, confidence=0.95)
    sample_mean_PDR_array1 =  np.append(sample_mean_PDR_array1, sample_mean1)
    sample_mean_PDR_array_gpsr1 =  np.append(sample_mean_PDR_array_gpsr1, sample_mean2)
    margin_of_error_array1 = np.append(margin_of_error_array1, margin_of_error1)
    margin_of_error_array_gpsr1 = np.append(margin_of_error_array_gpsr1, margin_of_error2)
    
# print(sample_mean_PDR_array1)
# print(sample_mean_PDR_array_gpsr1)

sample_mean_PDR_array2 = np.concatenate([sample_mean_PDR_array1, sample_mean_PDR_array])
sample_mean_PDR_array_gpsr2 =  np.concatenate([sample_mean_PDR_array_gpsr1, sample_mean_PDR_array_gpsr])
margin_of_error_array2 = np.concatenate([margin_of_error_array1, margin_of_error_array])
margin_of_error_array_gpsr2 = np.concatenate([margin_of_error_array_gpsr1, margin_of_error_array_gpsr])

# print(sample_mean_PDR_array_gpsr2)

plt.errorbar(x=sent_packets_intervals,y= sample_mean_PDR_array2,yerr=margin_of_error_array2,fmt='D',capsize=4, capthick=1, color='m', markersize=8,label='DSPR') #ecolor='c'
plt.errorbar(x=sent_packets_intervals,y= sample_mean_PDR_array_gpsr2,yerr=margin_of_error_array_gpsr2,fmt='o',capsize=4, capthick=1, color='b', markersize=8,label='GPSR')  #ecolor='r'
plt.yticks(range(8))
plt.xlabel('Packet sending Intervals in seconds')
plt.ylabel('Packet Delivery ratio')
# plt.title('Average Packet Delivery ratio using CI')
plt.legend()
plt.grid(True)
# plt.savefig('PDR_12_nodes_scenario_CI.pdf', format='pdf', dpi=1200)
# plt.savefig('PDR_NAC_scenario_CI.pdf', format='pdf', dpi=1200)
# plt.savefig('PDR_NAC_A2G_scenario_CI.pdf', format='pdf', dpi=1200)
plt.show()