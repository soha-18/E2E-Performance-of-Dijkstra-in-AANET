import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

def process_dataframe(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue'])
    df.reset_index(drop=True, inplace=True)
    index_values = list(range(len(df)))
    index_reset_values = [i // 371 for i in index_values]
    df['new_index'] = index_reset_values
    df = df[['new_index'] + [col for col in df.columns if col != 'new_index']]
    df = df.drop(columns=['run'])
    df = df.rename(columns={'new_index': 'run'})
    df['commRange'] = df['run'].apply(calculate_comm_range)
    df = df[['commRange'] + [col for col in df.columns if col != 'commRange']]
    df = df.rename(columns={'run': 'SimulationRun'})
    # df = df.drop(columns=['range'])
    return df

def process_receive_dataframe(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue'])
    df.reset_index(drop=True, inplace=True)
    df['run'] = df.index
    df['commRange'] = df['run'].apply(calculate_comm_range)
    df = df[['commRange'] + [col for col in df.columns if col != 'commRange']]
    df = df.rename(columns={'run': 'SimulationRun'})
    # df = df.drop(columns=['range'])
    return df

def process_receive_dataframe1(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue'])
    df.reset_index(drop=True, inplace=True)
    df['run'] = df.index
    df['commRange'] = df['run'].apply(calculate_comm_range1)
    df = df[['commRange'] + [col for col in df.columns if col != 'commRange']]
    df = df.rename(columns={'run': 'SimulationRun'})
    # df = df.drop(columns=['range'])
    return df

def process_dataframe_gpsr(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue'])
    df.reset_index(drop=True, inplace=True)
    df['new_index'] = (df.index // 371)
    df = df[['new_index'] + [col for col in df.columns if col != 'new_index']]
    df = df.drop(columns=['run'])
    df = df.rename(columns={'new_index': 'run'})
    df['commRange'] = df['run'].apply(calculate_comm_range)
    df = df[['commRange'] + [col for col in df.columns if col != 'commRange']]
    df = df.loc[:, ~df.columns.duplicated()]
   #  df = df.rename(columns={'run': 'SimulationRun'})
    return df

def process_dataframe_gpsr1(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue'])
    df.reset_index(drop=True, inplace=True)
    df['new_index'] = (df.index // 371)
    df = df[['new_index'] + [col for col in df.columns if col != 'new_index']]
    df = df.drop(columns=['run'])
    df = df.rename(columns={'new_index': 'run'})
    df['commRange'] = df['run'].apply(calculate_comm_range1)
    df = df[['commRange'] + [col for col in df.columns if col != 'commRange']]
    df = df.loc[:, ~df.columns.duplicated()]
   #  df = df.rename(columns={'run': 'SimulationRun'})
    return df

def calculate_comm_range(run_value):
    run_value = int(run_value)
    if 0 <= run_value <= 9:
        return 170
    elif 10 <= run_value <= 19:
        return 220
    elif 20 <= run_value <= 29:
        return 270
    elif 30 <= run_value <= 39:
        return 320
    else:
        return 370
    
def calculate_comm_range1(run_value):
    run_value = int(run_value)
    if 0 <= run_value <= 9:
        return 320
    else:
        return 370
    
#function to calculate confidence interval
def confidence_interval_t(data, confidence=0.95):
    data_array = 1.0 * np.array(data)
    degree_of_freedom = len(data_array) - 1
    sample_mean, sample_standard_error = np.mean(data_array), stats.sem(data_array)
    t = stats.t.ppf((1 + confidence) / 2., degree_of_freedom)
    margin_of_error = sample_standard_error * t
    Confidence_Interval = 1.0 * np.array([sample_mean - margin_of_error, sample_mean + margin_of_error])
    return sample_mean, Confidence_Interval, margin_of_error
    
    
packet_sent_file = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Comm_Range/packet_sent.csv'
packet_sent_file_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/packet_sent.csv'
packet_sent_file1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Comm_Range/packet_sent1.csv'
packet_sent_file_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/packet_sent1.csv'

packet_received_file = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Comm_Range/packet_received.csv'
packet_received_file_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/packet_received.csv'
packet_received_file1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Comm_Range/packet_received1.csv'
packet_received_file_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/packet_received1.csv'

sent_vector = pd.read_csv(packet_sent_file)
sent_vector_gpsr = pd.read_csv(packet_sent_file_gpsr)
sent_vector1 = pd.read_csv(packet_sent_file1)
sent_vector_gpsr1 = pd.read_csv(packet_sent_file_gpsr1)

receive_vector = pd.read_csv(packet_received_file)
receive_vector_gpsr = pd.read_csv(packet_received_file_gpsr)
receive_vector1 = pd.read_csv(packet_received_file1)
receive_vector_gpsr1 = pd.read_csv(packet_received_file_gpsr1)


column_to_check = 'name'
sent_vector = sent_vector.dropna(subset=[column_to_check])
receive_vector = receive_vector.dropna(subset=[column_to_check])
sent_vector_gpsr = sent_vector_gpsr.dropna(subset=[column_to_check])
receive_vector_gpsr = receive_vector_gpsr.dropna(subset=[column_to_check])
sent_vector1 = sent_vector1.dropna(subset=[column_to_check])
sent_vector_gpsr1 = sent_vector_gpsr1.dropna(subset=[column_to_check])
receive_vector1 = receive_vector1.dropna(subset=[column_to_check])
receive_vector_gpsr1 = receive_vector_gpsr1.dropna(subset=[column_to_check])

sent_vector = process_dataframe(sent_vector)
sent_vector_gpsr = process_dataframe(sent_vector_gpsr)

sent_vector1['run'] = sent_vector1['run'].str.extract(r'-(\d+)-')
sent_vector1 = sent_vector1[sent_vector1['type'] != 'attr']
sent_vector1 = sent_vector1.drop(columns=['name','type','module', 'attrname', 'attrvalue'])
sent_vector1.reset_index(drop=True, inplace=True)
index_values = list(range(len(sent_vector1)))
index_reset_values = [i // 371 for i in index_values]
sent_vector1['new_index'] = index_reset_values
sent_vector1 = sent_vector1[['new_index'] + [col for col in sent_vector1.columns if col != 'new_index']]
sent_vector1 = sent_vector1.drop(columns=['run'])
sent_vector1 = sent_vector1.rename(columns={'new_index': 'run'})
sent_vector1.loc[(sent_vector1['run'] >= 0) & (sent_vector1['run'] <= 9), 'run'] += 40
sent_vector1['commRange'] = 370
sent_vector1 = sent_vector1[['commRange'] + [col for col in sent_vector1.columns if col != 'commRange']]
sent_vector1 = sent_vector1.rename(columns={'run': 'SimulationRun'})

sent_vector_gpsr1 = process_dataframe_gpsr1(sent_vector_gpsr1)
sent_vector_gpsr1 = sent_vector_gpsr1.rename(columns={'run': 'SimulationRun'})
sent_vector_gpsr1['SimulationRun'] = sent_vector_gpsr1['SimulationRun'].astype(int)
sent_vector_gpsr1.loc[(sent_vector_gpsr1['SimulationRun'] >= 0) & (sent_vector_gpsr1['SimulationRun'] <= 19), 'SimulationRun'] += 30


receive_vector = process_receive_dataframe(receive_vector)
receive_vector_gpsr = process_receive_dataframe(receive_vector_gpsr)

receive_vector1['run'] = receive_vector1['run'].str.extract(r'-(\d+)-')
receive_vector1 = receive_vector1[receive_vector1['type'] != 'attr']
receive_vector1 = receive_vector1.drop(columns=['name','type','module', 'attrname', 'attrvalue'])
receive_vector1['commRange'] = 370
receive_vector1 = receive_vector1[['commRange'] + [col for col in receive_vector1.columns if col != 'commRange']]
receive_vector1['run'] = receive_vector1['run'].astype(int)
receive_vector1.loc[(receive_vector1['run'] >= 0) & (receive_vector1['run'] <= 9), 'run'] += 40
receive_vector1 = receive_vector1.rename(columns={'run': 'SimulationRun'})

receive_vector_gpsr1 = process_receive_dataframe1(receive_vector_gpsr1)
receive_vector_gpsr1['SimulationRun'] = receive_vector_gpsr1['SimulationRun'].astype(int)
receive_vector_gpsr1.loc[(receive_vector_gpsr1['SimulationRun'] >= 0) & (receive_vector_gpsr1['SimulationRun'] <= 19), 'SimulationRun'] += 30


Packet_sent = pd.concat([sent_vector, sent_vector1])
Packet_sent_gpsr = pd.concat([sent_vector_gpsr, sent_vector_gpsr1])

Packet_received = pd.concat([receive_vector, receive_vector1])
Packet_received_gpsr = pd.concat([receive_vector_gpsr, receive_vector_gpsr1])


Packet_sent = Packet_sent.rename(columns={'value': 'Packet_Sent'})
Packet_sent = Packet_sent.groupby(['commRange', 'SimulationRun'])['Packet_Sent'].sum().reset_index()
Packet_sent_gpsr = Packet_sent_gpsr.rename(columns={'value': 'Packet_Sent'})
Packet_sent_gpsr = Packet_sent_gpsr.groupby(['commRange', 'SimulationRun'])['Packet_Sent'].sum().reset_index()
# print(Packet_sent)
Packet_received = Packet_received.rename(columns={'value': 'Packet_Received'})
Packet_received_gpsr = Packet_received_gpsr.rename(columns={'value': 'Packet_Received'})
# print(receive_vector_gpsr)

PDR_df = pd.merge(Packet_sent, Packet_received, on=['commRange', 'SimulationRun'], how='inner')
PDR_df_gpsr = pd.merge(Packet_sent_gpsr, Packet_received_gpsr, on=['commRange', 'SimulationRun'], how='inner')
PDR_df['PDR'] = PDR_df['Packet_Received']/PDR_df['Packet_Sent']
PDR_df_gpsr['PDR'] = PDR_df_gpsr['Packet_Received']/PDR_df_gpsr['Packet_Sent']
PDR_df = PDR_df.drop(columns=['Packet_Received', 'Packet_Sent'])
PDR_df_gpsr = PDR_df_gpsr.drop(columns=['Packet_Received', 'Packet_Sent'])
# print(PDR_df)
# print(PDR_df_gpsr)


commRange_values = Packet_received['commRange'].unique()

Communication_range = ['170', '220', '270', '320', '370']

sample_mean_PDR_array = []
margin_of_error_array = []
sample_mean_PDR_array_gpsr = []
margin_of_error_array_gpsr = []

for val in commRange_values:
    group1 = PDR_df[PDR_df['commRange'] == val]
    group2 = PDR_df_gpsr[PDR_df_gpsr['commRange'] == val]
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

plt.errorbar(x=Communication_range,y= sample_mean_PDR_array,yerr=margin_of_error_array,fmt='o',capsize=4, capthick=1, color='m', markersize=8,label='DSPR') #ecolor='c'
plt.errorbar(x=Communication_range,y= sample_mean_PDR_array_gpsr,yerr=margin_of_error_array_gpsr,fmt='D',capsize=4, capthick=1, color='b', markersize=8,label='GPSR')  #ecolor='r'


plt.xlabel('Communication Range in kilometres')
plt.ylabel('Packet Delivery ratio')
plt.legend()
plt.grid(True)
plt.savefig('PDR_NAC_A2G_scenario_commRange_CI.pdf', format='pdf', dpi=1200)
plt.show()