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
communication_range = [170000, 220000, 270000, 320000, 370000]
groundStation_range = 370400
adjacency_matrices = []

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
    df = df.rename(columns={'run': 'SimulationRun'})
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
    df = df.rename(columns={'run': 'SimulationRun'})
    return df

def process_dataframe(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
    df.reset_index(drop=True, inplace=True)
    df['run'] = df.index
    df['commRange'] = df['run'].apply(calculate_comm_range)
    df = df[['commRange'] + [col for col in df.columns if col != 'commRange']]
    df = df.rename(columns={'run': 'SimulationRun'})
    # df = df.drop(columns=['range'])
    return df

def process_dataframe1(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
    df.reset_index(drop=True, inplace=True)
    df['run'] = df.index
    df['commRange'] = df['run'].apply(calculate_comm_range1)
    df = df[['commRange'] + [col for col in df.columns if col != 'commRange']]
    df = df.rename(columns={'run': 'SimulationRun'})
    # df = df.drop(columns=['range'])
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

unique_timestamps = node_mobility['timestamps'].unique()
results = []
for timestamp in unique_timestamps:
    nodes_at_timestamp = positions_at_timestamp[timestamp]
    num_nodes = len(nodes_at_timestamp)
    # print(num_nodes)
    destIdx = num_nodes - 1
    adjacency_matrix = np.zeros((num_nodes, num_nodes))  
    for h in communication_range:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if i != j:
                   distance = euclidean_distance(nodes_at_timestamp[i][1], nodes_at_timestamp[j][1])
                   modified_range = groundStation_range if i == destIdx or j == destIdx else h
                   if distance <= modified_range:
                      adjacency_matrix[i, j] = 1
                      adjacency_matrix[j, i] = 1
        adjacency_matrices.append(adjacency_matrix)
 
        shortest_distances = []
        for source in range(num_nodes):
            dist = dijkstra_priority_queue(adjacency_matrix, source, destIdx)
            shortest_distances.append({'source': source, 'hop_count': dist})
        
        for node_data in shortest_distances:
            results.append({
                'CommunicationRange': h,
                'timestamps': timestamp,
                'new_index': node_data['source'],
                'dest': destIdx,
                'hop_count': node_data['hop_count']
            })

# # Create a DataFrame from the results
shortest_path = pd.DataFrame(results)
shortest_path['commRange'] = shortest_path['CommunicationRange']/ 1000
shortest_path['commRange'] = shortest_path['commRange'].astype('int16')
shortest_path = shortest_path[['commRange'] + [col for col in shortest_path.columns if col != 'commRange']]


mapped_df = node_mobility.copy()
mapped_df = mapped_df.drop(columns=['x', 'y', 'z'])
shortest_path_df = pd.merge(shortest_path, mapped_df, on=['timestamps', 'new_index'], how='inner')
shortest_path_df['dest'] = shortest_path_df.groupby('CommunicationRange')['node'].transform('last')
shortest_path_df = shortest_path_df.drop(columns=['new_index', 'CommunicationRange'])
shortest_path_df = shortest_path_df[['commRange', 'timestamps', 'node', 'dest', 'hop_count']]
shortest_path_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# print(shortest_path_df)

# print(shortest_path_df[shortest_path_df['commRange'] == 220])

# column_name = 'node'
# subset = shortest_path_df[shortest_path_df['commRange'] == 220]
# column_values = subset[column_name].tolist()
# print(column_values)


file_path_hopCount = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/hopCount_vector.csv'
send_packet_file_path_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/packetId_sent_vector.csv'
receive_packet_file_path_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/packetId_received_vector.csv'
file_path_hopCount1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/hopCount_vector1.csv'
send_packet_file_path_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/packetId_sent_vector1.csv'
receive_packet_file_path_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/packetId_received_vector1.csv'

df = pd.read_csv(file_path_hopCount)
df1 = pd.read_csv(send_packet_file_path_gpsr)
df2 = pd.read_csv(receive_packet_file_path_gpsr)

df3 = pd.read_csv(file_path_hopCount1)
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
df1 = process_dataframe_gpsr(df1)
df2 = process_dataframe(df2)

df3 = process_dataframe1(df3)
df3.loc[(df3['SimulationRun'] >= 0) & (df3['SimulationRun'] <= 19), 'SimulationRun'] += 30
df4 = process_dataframe_gpsr1(df4)
df4.loc[(df4['SimulationRun'] >= 0) & (df4['SimulationRun'] <= 19), 'SimulationRun'] += 30
df5 = process_dataframe1(df5)
df5.loc[(df5['SimulationRun'] >= 0) & (df5['SimulationRun'] <= 19), 'SimulationRun'] += 30

Packet_sent_file = pd.concat([df1, df4])
Packet_receive_file = pd.concat([df2, df5])
hopCount_file = pd.concat([df, df3])

# print(hopCount_file)

Packet_sent_file = Packet_sent_file.rename(columns={'vectime': 'PacketSentTime'})
Packet_sent_file = Packet_sent_file.rename(columns={'run': 'SimulationRun'})
Packet_sent_file = Packet_sent_file.rename(columns={'vecvalue': 'packetID'})
Packet_sent_file['packetID'] = Packet_sent_file['packetID'].str.split()
Packet_sent_file['PacketSentTime'] = Packet_sent_file['PacketSentTime'].str.split()

Packet_sent_file = pd.DataFrame({
    'commRange': Packet_sent_file['commRange'].repeat(Packet_sent_file['packetID'].apply(len)),
    'SimulationRun': Packet_sent_file['SimulationRun'].repeat(Packet_sent_file['packetID'].apply(len)),
    'packetID': [x for sublist in Packet_sent_file['packetID'] for x in sublist],
    'PacketSentTime': [x for sublist in Packet_sent_file['PacketSentTime'] for x in sublist]
})
Packet_sent_file['commRange'] = pd.to_numeric(Packet_sent_file['commRange'], errors='coerce')
Packet_sent_file['SimulationRun'] = pd.to_numeric(Packet_sent_file['SimulationRun'], errors='coerce')
Packet_sent_file['packetID'] = pd.to_numeric(Packet_sent_file['packetID'], errors='coerce')
Packet_sent_file['PacketSentTime'] = pd.to_numeric(Packet_sent_file['PacketSentTime'], errors='coerce')
Packet_sent_file.reset_index(drop=True, inplace=True)

filtered_packet_sent_file = Packet_sent_file[(Packet_sent_file['PacketSentTime'] >= 200.0) & (Packet_sent_file['PacketSentTime'] <= 1500.0)]
# print(filtered_packet_sent_file)

merged_df = Packet_receive_file.merge(hopCount_file, on=['commRange', 'vectime'], suffixes=('_df2', '_df'))
merged_df = merged_df.drop(columns=['SimulationRun_df'])
merged_df = merged_df.rename(columns={'vecvalue_df': 'GPSRHopCount'})
merged_df = merged_df.rename(columns={'SimulationRun_df2': 'SimulationRun'})
merged_df = merged_df.rename(columns={'vecvalue_df2': 'packetID'})
merged_df['vectime'] = merged_df['vectime'].str.split()
merged_df['packetID'] = merged_df['packetID'].str.split()
merged_df['GPSRHopCount'] = merged_df['GPSRHopCount'].str.split()

merged_df = pd.DataFrame({
    'commRange': merged_df['commRange'].repeat(merged_df['packetID'].apply(len)),
    'SimulationRun': merged_df['SimulationRun'].repeat(merged_df['packetID'].apply(len)),
    'vectime': [x for sublist in merged_df['vectime'] for x in sublist],
    'packetID': [x for sublist in merged_df['packetID'] for x in sublist],
    'GPSRHopCount': [x for sublist in merged_df['GPSRHopCount'] for x in sublist]
})
merged_df['commRange'] = pd.to_numeric(merged_df['commRange'], errors='coerce')
merged_df['SimulationRun'] = pd.to_numeric(merged_df['SimulationRun'], errors='coerce')
merged_df['vectime'] = pd.to_numeric(merged_df['vectime'], errors='coerce')
merged_df['packetID'] = pd.to_numeric(merged_df['packetID'], errors='coerce')
merged_df['GPSRHopCount'] = pd.to_numeric(merged_df['GPSRHopCount'], errors='coerce')
merged_df.reset_index(drop=True, inplace=True)
filtered_merged_df = merged_df[(merged_df['vectime'] >= 200.0) & (merged_df['vectime'] <= 1500.0)]
filtered_merged_df = filtered_merged_df.drop(columns=['vectime'])
# print(filtered_merged_df)


result_df = pd.merge(filtered_merged_df, filtered_packet_sent_file, on=['commRange', 'SimulationRun', 'packetID'], how='inner')
# filtered_df = result_df[(result_df['PacketSentTime'] >= 200.0) & (result_df['PacketSentTime'] <= 1500.0)]
result_df = result_df.drop_duplicates()
result_df.reset_index(drop=True, inplace=True)
# print(result_df)

commRange_values = filtered_merged_df['commRange'].unique()
time_intervals = list(range(0, 1801, 60))
# #time_intervals = [0, 1000]
# # Map the sending_time in result_df to the nearest timestamps in shortest_path
result_df['timestamps_mapped'] = pd.cut(result_df['PacketSentTime'], bins=time_intervals, labels=time_intervals[:-1])
filtered_df = result_df.rename(columns={'timestamps_mapped': 'timestamps'})
filtered_df = filtered_df[['timestamps'] + [col for col in filtered_df.columns if col != 'timestamps']]
filtered_df['packetID_int'] = filtered_df['packetID'].astype('int16')
filtered_df = filtered_df.rename(columns={'packetID_int': 'node'})
final_df = pd.merge(filtered_df, shortest_path_df, on=['commRange','timestamps', 'node'], how='inner')
final_df = final_df.rename(columns={'hop_count': 'shortestPathHop'})
final_df['Hop Stretch Factor'] = final_df['GPSRHopCount'] / final_df['shortestPathHop']
final_df = final_df.drop(columns=['node', 'dest'])
final_df = final_df.drop_duplicates()
final_df.reset_index(drop=True, inplace=True)


Communication_range = ['170', '220', '270', '320', '370']

sample_mean_Hop_Stretch = []
margin_of_error_array= []

for val in commRange_values:
    group = final_df[final_df['commRange'] == val]
    hop_stretch_factor = group['Hop Stretch Factor']
    hop_stretch_factor = hop_stretch_factor.dropna()
    sample_mean, confidence_Interval, margin_of_error = confidence_interval_t(hop_stretch_factor, confidence=0.95)
    sample_mean_Hop_Stretch =  np.append(sample_mean_Hop_Stretch, sample_mean)
    margin_of_error_array = np.append(margin_of_error_array, margin_of_error)

# print(sample_mean_Hop_Stretch)

# # CI Plot
plt.errorbar(x = Communication_range, y= sample_mean_Hop_Stretch, yerr=margin_of_error_array, fmt='o', capsize=4, capthick=1, label='Hop Stretch Factor')
plt.xlabel('Communication Range in kilometres')
plt.ylabel('Hop Stretch Factor')
# plt.legend()
plt.grid(True)
# plt.savefig('Hop_stretch_factor_NAC_A2G_scenario_commRange_CI.pdf', format='pdf', dpi=1200)
plt.show()