import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D
import numpy as np
import heapq
from scipy import stats

no_simulation_runs = 10
num_sets = 5

timestamps = []
nodes = []
x_values = []
y_values = []
z_values = []
positions_at_timestamp = {}
communication_range = 370400
#communication_range = 1.42
adjacency_matrices = []

def process_dataframe(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
    df.reset_index(drop=True, inplace=True)
    df['sendInterval'] = df['run'].apply(calculate_send_interval)
    df = df[['sendInterval'] + [col for col in df.columns if col != 'sendInterval']]
    return df

def process_dataframe1(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
    df.reset_index(drop=True, inplace=True)
    df['sendInterval'] = df['run'].apply(calculate_send_interval1)
    df = df[['sendInterval'] + [col for col in df.columns if col != 'sendInterval']]
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
        return 90
    else:
        return 170
        # return 10000
        
def dijkstra_priority_queue(adj_matrix, source, destination):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    hop_count = [float('inf')] * num_nodes

    hop_count[source] = 0

    priority_queue = [(0, source)]

    while priority_queue:
        _, current_node = heapq.heappop(priority_queue)

        if visited[current_node]:
            continue
        visited[current_node] = True

        for neighbor in range(num_nodes):
            if not visited[neighbor] and adj_matrix[current_node][neighbor]:
                new_hop_count = hop_count[current_node] + adj_matrix[current_node][neighbor]
                if new_hop_count < hop_count[neighbor]:
                    hop_count[neighbor] = new_hop_count
                    heapq.heappush(priority_queue, (new_hop_count, neighbor))

    return hop_count[destination]

#function to calculate confidence interval
def confidence_interval_t(data, confidence=0.95):
    data_array = 1.0 * np.array(data)
    degree_of_freedom = len(data_array) - 1
    sample_mean, sample_standard_error = np.mean(data_array), stats.sem(data_array)
    t = stats.t.ppf((1 + confidence) / 2., degree_of_freedom)
    margin_of_error = sample_standard_error * t
    Confidence_Interval = 1.0 * np.array([sample_mean - margin_of_error, sample_mean + margin_of_error])
    return sample_mean, Confidence_Interval, margin_of_error

def euclidean_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

# Open and read the text file
with open('NATselB_2019_mobility.txt', 'r') as file:
#with open('mobility.txt', 'r') as file:
    lines = file.readlines()

node = 0

for line in lines:
    values = line.split()

    # Check if there are at least 4 values on each line (timestamp and (x, y, z) coordinates)
    if len(values) >= 4:
        for i in range(0, len(values), 4):
            timestamp = int(values[i])
            x = float(values[i + 1])
            y = float(values[i + 2])
            z = float(values[i + 3])

            # Append values to their respective lists
            timestamps.append(timestamp)
            nodes.append(node)
            x_values.append(x)
            y_values.append(y)
            z_values.append(z)

            # Update the dictionary with node positions at this timestamp
            if timestamp not in positions_at_timestamp:
                positions_at_timestamp[timestamp] = []
            positions_at_timestamp[timestamp].append((node, (x, y, z)))

        node += 1

# # Create a DataFrame from the lists
node_mobility = pd.DataFrame({'timestamps': timestamps,'node': nodes,'x': x_values,'y': y_values,'z': z_values})
# timestamp_groups = node_mobility.groupby('timestamps')
node_mobility['new_index'] = node_mobility.groupby('timestamps').cumcount()
node_mobility = pd.concat([node_mobility.iloc[:, :1], node_mobility['new_index'], node_mobility.iloc[:, 1:]], axis=1)
node_mobility = node_mobility.loc[:, ~node_mobility.columns.duplicated()]

for timestamp in sorted(positions_at_timestamp.keys()):
    nodes_at_timestamp = positions_at_timestamp[timestamp]
    num_nodes = len(nodes_at_timestamp)
    # print(num_nodes)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i != j:
               distance = euclidean_distance(nodes_at_timestamp[i][1], nodes_at_timestamp[j][1])
               if distance <= communication_range:
                  adjacency_matrix[i, j] = 1
                  adjacency_matrix[j, i] = 1
                        
    adjacency_matrices.append(adjacency_matrix)
# print(adjacency_matrices[0])
dfs = []
    
for timestamp, adjacency_matrix in zip(timestamps, adjacency_matrices):
    destination_node = len(adjacency_matrix) - 1
    hop_count_data = []
    
    for source_node in range(len(adjacency_matrix)):
        hop_count = dijkstra_priority_queue(adjacency_matrix, source_node, destination_node)
        results = {'timestamps': timestamp, 'new_index': source_node, 'dest' : destination_node, 'hop_count': hop_count}
        hop_count_data.append(results)
        
    df = pd.DataFrame(hop_count_data)
    dfs.append(df)
shortest_path = pd.concat(dfs)

mapped_df = node_mobility.copy()
mapped_df = mapped_df.drop(columns=['x', 'y', 'z'])
# print(mapped_df[mapped_df['timestamps'] == 180]) #correct
shortest_path_df = pd.merge(mapped_df, shortest_path, on=['timestamps', 'new_index'], how='inner')
shortest_path_df['dest'] = shortest_path_df.groupby('timestamps')['node'].transform('last')
shortest_path_df = shortest_path_df.drop(columns=['new_index'])


# file_path_gpsr = 'E:/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/GPSR/hopCount_vector.csv'
file_path_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/hopCount_vector.csv'
# send_packet_file_path_gpsr = 'E:/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/GPSR/packetId_sent_vector.csv'
send_packet_file_path_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/packetId_sent_vector.csv'
# receive_packet_file_path_gpsr = 'E:/Study materials/Thesis/Thesis_Code/Results/12_nodes_scenario/GPSR/packetId_received_vector.csv'
receive_packet_file_path_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/packetId_received_vector.csv'

file_path_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/hopCount_vector1.csv'
send_packet_file_path_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/packetId_sent_vector1.csv'
receive_packet_file_path_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/packetId_received_vector1.csv'

df = pd.read_csv(file_path_gpsr)
df1 = pd.read_csv(send_packet_file_path_gpsr)
df2 = pd.read_csv(receive_packet_file_path_gpsr)

df3 = pd.read_csv(file_path_gpsr1)
df4 = pd.read_csv(send_packet_file_path_gpsr1)
df5 = pd.read_csv(receive_packet_file_path_gpsr1)

column_to_check = 'name'
df = df.dropna(subset=[column_to_check])
df1 = df1.dropna(subset=[column_to_check])
df2 = df2.dropna(subset=[column_to_check])

df3 = df3.dropna(subset=[column_to_check])
df4 = df4.dropna(subset=[column_to_check])
df5 = df5.dropna(subset=[column_to_check])

df = process_dataframe(df)
df1 = process_dataframe(df1)
df2 = process_dataframe(df2)

df3 = process_dataframe1(df3)
df4 = process_dataframe1(df4)
df5 = process_dataframe1(df5)


df1 = df1.rename(columns={'vectime': 'PacketSentTime'})
df1 = df1.rename(columns={'run': 'SimulationRun'})
df1 = df1.rename(columns={'vecvalue': 'packetID'})
df1['packetID'] = df1['packetID'].str.split()
df1['PacketSentTime'] = df1['PacketSentTime'].str.split()

df1 = pd.DataFrame({
    'sendInterval': df1['sendInterval'].repeat(df1['packetID'].apply(len)),
    'SimulationRun': df1['SimulationRun'].repeat(df1['packetID'].apply(len)),
    'packetID': [x for sublist in df1['packetID'] for x in sublist],
    'PacketSentTime': [x for sublist in df1['PacketSentTime'] for x in sublist]
})
df1['sendInterval'] = pd.to_numeric(df1['sendInterval'], errors='coerce')
df1['SimulationRun'] = pd.to_numeric(df1['SimulationRun'], errors='coerce')
df1['packetID'] = pd.to_numeric(df1['packetID'], errors='coerce')
df1['PacketSentTime'] = pd.to_numeric(df1['PacketSentTime'], errors='coerce')
filtered_df1 = df1[(df1['PacketSentTime'] >= 200.0) & (df1['PacketSentTime'] <= 1500.0)]
filtered_df1.reset_index(drop=True, inplace=True)

df4 = df4.rename(columns={'vectime': 'PacketSentTime'})
df4 = df4.rename(columns={'run': 'SimulationRun'})
df4 = df4.rename(columns={'vecvalue': 'packetID'})
df4['packetID'] = df4['packetID'].str.split()
df4['PacketSentTime'] = df4['PacketSentTime'].str.split()

df4 = pd.DataFrame({
    'sendInterval': df4['sendInterval'].repeat(df4['packetID'].apply(len)),
    'SimulationRun': df4['SimulationRun'].repeat(df4['packetID'].apply(len)),
    'packetID': [x for sublist in df4['packetID'] for x in sublist],
    'PacketSentTime': [x for sublist in df4['PacketSentTime'] for x in sublist]
})
df4['sendInterval'] = pd.to_numeric(df4['sendInterval'], errors='coerce')
df4['SimulationRun'] = pd.to_numeric(df4['SimulationRun'], errors='coerce')
df4['packetID'] = pd.to_numeric(df4['packetID'], errors='coerce')
df4['PacketSentTime'] = pd.to_numeric(df4['PacketSentTime'], errors='coerce')
filtered_df4 = df4[(df4['PacketSentTime'] >= 200.0) & (df4['PacketSentTime'] <= 1500.0)]
filtered_df4.reset_index(drop=True, inplace=True)


merged_df = df2.merge(df, on=['sendInterval', 'vectime'], suffixes=('_df2', '_df'))
merged_df = merged_df.drop(columns=['run_df'])
merged_df = merged_df.rename(columns={'vecvalue_df': 'GPSRHopCount'})
merged_df = merged_df.rename(columns={'run_df2': 'SimulationRun'})
merged_df = merged_df.rename(columns={'vecvalue_df2': 'packetID'})
merged_df['vectime'] = merged_df['vectime'].str.split()
merged_df['packetID'] = merged_df['packetID'].str.split()
merged_df['GPSRHopCount'] = merged_df['GPSRHopCount'].str.split()


merged_df = pd.DataFrame({
    'sendInterval': merged_df['sendInterval'].repeat(merged_df['packetID'].apply(len)),
    'SimulationRun': merged_df['SimulationRun'].repeat(merged_df['packetID'].apply(len)),
    'vectime': [x for sublist in merged_df['vectime'] for x in sublist],
    'packetID': [x for sublist in merged_df['packetID'] for x in sublist],
    'GPSRHopCount': [x for sublist in merged_df['GPSRHopCount'] for x in sublist]
})
merged_df['sendInterval'] = pd.to_numeric(merged_df['sendInterval'], errors='coerce')
merged_df['SimulationRun'] = pd.to_numeric(merged_df['SimulationRun'], errors='coerce')
merged_df['vectime'] = pd.to_numeric(merged_df['vectime'], errors='coerce')
merged_df['packetID'] = pd.to_numeric(merged_df['packetID'], errors='coerce')
merged_df['GPSRHopCount'] = pd.to_numeric(merged_df['GPSRHopCount'], errors='coerce')
merged_df.reset_index(drop=True, inplace=True)
filtered_merged_df = merged_df[(merged_df['vectime'] >= 200.0) & (merged_df['vectime'] <= 1500.0)]
filtered_merged_df = filtered_merged_df.drop(columns=['vectime'])
# print(merged_df)


merged_df1 = df5.merge(df3, on=['sendInterval', 'vectime'], suffixes=('_df2', '_df'))
merged_df1 = merged_df1.drop(columns=['run_df'])
merged_df1 = merged_df1.rename(columns={'vecvalue_df': 'GPSRHopCount'})
merged_df1 = merged_df1.rename(columns={'run_df2': 'SimulationRun'})
merged_df1 = merged_df1.rename(columns={'vecvalue_df2': 'packetID'})
merged_df1['vectime'] = merged_df1['vectime'].str.split()
merged_df1['packetID'] = merged_df1['packetID'].str.split()
merged_df1['GPSRHopCount'] = merged_df1['GPSRHopCount'].str.split()


merged_df1 = pd.DataFrame({
    'sendInterval': merged_df1['sendInterval'].repeat(merged_df1['packetID'].apply(len)),
    'SimulationRun': merged_df1['SimulationRun'].repeat(merged_df1['packetID'].apply(len)),
    'vectime': [x for sublist in merged_df1['vectime'] for x in sublist],
    'packetID': [x for sublist in merged_df1['packetID'] for x in sublist],
    'GPSRHopCount': [x for sublist in merged_df1['GPSRHopCount'] for x in sublist]
})
merged_df1['sendInterval'] = pd.to_numeric(merged_df1['sendInterval'], errors='coerce')
merged_df1['SimulationRun'] = pd.to_numeric(merged_df1['SimulationRun'], errors='coerce')
merged_df1['vectime'] = pd.to_numeric(merged_df1['vectime'], errors='coerce')
merged_df1['packetID'] = pd.to_numeric(merged_df1['packetID'], errors='coerce')
merged_df1['GPSRHopCount'] = pd.to_numeric(merged_df1['GPSRHopCount'], errors='coerce')
merged_df1.reset_index(drop=True, inplace=True)
filtered_merged_df1 = merged_df1[(merged_df1['vectime'] >= 200.0) & (merged_df1['vectime'] <= 1500.0)]
filtered_merged_df1 = filtered_merged_df1.drop(columns=['vectime'])


result_df = pd.merge(filtered_merged_df, filtered_df1, on=['sendInterval', 'SimulationRun', 'packetID'], how='inner')
# filtered_df = result_df[(result_df['PacketSentTime'] >= 200.0) & (result_df['PacketSentTime'] <= 1500.0)]
result_df = result_df.drop_duplicates()
result_df.reset_index(drop=True, inplace=True)
# print(result_df)

result_df1 = pd.merge(filtered_merged_df1, filtered_df4, on=['sendInterval', 'SimulationRun', 'packetID'], how='inner')
# filtered_df1 = result_df1[(result_df1['PacketSentTime'] >= 200.0) & (result_df1['PacketSentTime'] <= 1500.0)]
result_df1 = result_df1.drop_duplicates()
result_df1.reset_index(drop=True, inplace=True)

time_intervals = list(range(0, 1801, 60))
# #time_intervals = [0, 1000]
# # Map the sending_time in result_df to the nearest timestamps in shortest_path
result_df['timestamps_mapped'] = pd.cut(result_df['PacketSentTime'], bins=time_intervals, labels=time_intervals[:-1])
filtered_df = result_df.rename(columns={'timestamps_mapped': 'timestamps'})
filtered_df = filtered_df[['timestamps'] + [col for col in filtered_df.columns if col != 'timestamps']]
filtered_df['packetID_int'] = filtered_df['packetID'].astype('int16')
filtered_df = filtered_df.rename(columns={'packetID_int': 'node'})
final_df = pd.merge(filtered_df, shortest_path_df, on=['timestamps', 'node'], how='inner')
final_df = final_df.rename(columns={'hop_count': 'shortestPathHop'})
final_df['Hop Stretch Factor'] = final_df['GPSRHopCount'] / final_df['shortestPathHop']
final_df = final_df[final_df['Hop Stretch Factor'] >= 1]
final_df = final_df.drop(columns=['node', 'dest'])
final_df = final_df.drop_duplicates()
final_df.reset_index(drop=True, inplace=True)

result_df1['timestamps_mapped'] = pd.cut(result_df1['PacketSentTime'], bins=time_intervals, labels=time_intervals[:-1])
result_df1 = result_df1.rename(columns={'timestamps_mapped': 'timestamps'})
result_df1 = result_df1[['timestamps'] + [col for col in result_df1.columns if col != 'timestamps']]
result_df1['packetID_int'] = result_df1['packetID'].astype('int16')
result_df1 = result_df1.rename(columns={'packetID_int': 'node'})
final_df1 = pd.merge(result_df1, shortest_path_df, on=['timestamps', 'node'], how='inner')
final_df1 = final_df1.rename(columns={'hop_count': 'shortestPathHop'})
final_df1['Hop Stretch Factor'] = final_df1['GPSRHopCount'] / final_df1['shortestPathHop']
final_df1 = final_df1[final_df1['Hop Stretch Factor'] >= 1]
final_df1 = final_df1.drop(columns=['node', 'dest'])
final_df1 = final_df1.drop_duplicates()
final_df1.reset_index(drop=True, inplace=True)


sendInterval_values = final_df['sendInterval'].unique()
sendInterval_values = sorted(sendInterval_values)
sendInterval_values1 = final_df1['sendInterval'].unique()
# print(sendInterval_values) #correct
# # sent_packets_intervals = ['1ms', '10ms', '100ms', '1000ms', '10000ms']
sent_packets_intervals = ['50','70','90', '110', '130', '150', '170']
sample_mean_Hop_Stretch = []
margin_of_error_array= []

sample_mean_Hop_Stretch1 = []
margin_of_error_array1= []

for timestamp in sendInterval_values:
    group = final_df[final_df['sendInterval'] == timestamp]
    hop_stretch_factor = group['Hop Stretch Factor']
    hop_stretch_factor = hop_stretch_factor.dropna()
    sample_mean, confidence_Interval, margin_of_error = confidence_interval_t(hop_stretch_factor, confidence=0.95)
    sample_mean_Hop_Stretch =  np.append(sample_mean_Hop_Stretch, sample_mean)
    margin_of_error_array = np.append(margin_of_error_array, margin_of_error)

for timestamp in sendInterval_values1:
    group = final_df1[final_df1['sendInterval'] == timestamp]
    hop_stretch_factor = group['Hop Stretch Factor']
    hop_stretch_factor = hop_stretch_factor.dropna()
    sample_mean, confidence_Interval, margin_of_error = confidence_interval_t(hop_stretch_factor, confidence=0.95)
    sample_mean_Hop_Stretch1 =  np.append(sample_mean_Hop_Stretch1, sample_mean)
    margin_of_error_array1 = np.append(margin_of_error_array1, margin_of_error)
# print(sample_mean_Hop_Stretch)

sample_mean_Hop_Stretch2 = np.concatenate([sample_mean_Hop_Stretch1, sample_mean_Hop_Stretch])
margin_of_error_array2 = np.concatenate([margin_of_error_array1, margin_of_error_array])

# print(sample_mean_Hop_Stretch2)
# print(margin_of_error_array2)

# # CI Plot
plt.errorbar(x = sent_packets_intervals, y= sample_mean_Hop_Stretch2, yerr=margin_of_error_array2, fmt='o', capsize=4, capthick=1, label='Hop Stretch Factor')
plt.xlabel('Packet Sending Intervals in seconds')
plt.ylabel('Hop Stretch Factor')
# plt.title('Average Hop Stretch Factor using CI')
# plt.legend()
plt.grid(True)
# plt.savefig('Hop_stretch_factor_NAC_A2G_scenario_CI.pdf', format='pdf', dpi=1200)
plt.show()