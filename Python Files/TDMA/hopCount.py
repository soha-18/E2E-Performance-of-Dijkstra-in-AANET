import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

no_simulation_runs = 10


# function to read csv file 
def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        data_vector = data[(data.type == 'vector')].vecvalue.to_numpy()  
        return data_vector
    
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None
    
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

#function to pre-process data and convert into usable format
def clean_data(dataset):
    dataset_array = [list(map(int, line.split())) for line in dataset]
    num_columns = max(len(row) for row in dataset_array)
    #print(num_columns)
    dataset_array = [row + [0] * (num_columns - len(row)) for row in dataset_array]
    return dataset_array


file_path = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/hopCount_vector.csv'
file_path_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/hopCount_vector.csv'

file_path1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Abstract TDMA/hopCount_vector1.csv'
file_path_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Abstract TDMA/hopCount_vector1.csv'

hopCount_vector = read_csv_file(file_path)
hopCount_vector_gpsr = read_csv_file(file_path_gpsr)
#print(len(hopCount_vector)) #50
hopCount_vector1 = read_csv_file(file_path1)
hopCount_vector_gpsr1 = read_csv_file(file_path_gpsr1)

hopCount_vector = hopCount_vector.reshape(5,no_simulation_runs)
hopCount_vector_gpsr = hopCount_vector_gpsr.reshape(5,no_simulation_runs)
# print(len(hopCount_vector[0])) #10

hopCount_vector1 = hopCount_vector1.reshape(2,no_simulation_runs)
hopCount_vector_gpsr1 = hopCount_vector_gpsr1.reshape(2,no_simulation_runs)

dataset1 = hopCount_vector_gpsr[0]
dataset2 = hopCount_vector_gpsr[1]
dataset3 = hopCount_vector_gpsr[2]
dataset4 = hopCount_vector_gpsr[3]
dataset5 = hopCount_vector_gpsr[4]

dataset6 = hopCount_vector_gpsr1[0]
dataset7 = hopCount_vector_gpsr1[1]


hopCount_array1 = clean_data(dataset1)
hopCount_array2 = clean_data(dataset2)
hopCount_array3 = clean_data(dataset3)
hopCount_array4 = clean_data(dataset4)
hopCount_array5 = clean_data(dataset5)

hopCount_array6 = clean_data(dataset6)
hopCount_array7 = clean_data(dataset7)

sorted_data1, cdf_data1 = calculate_cdf(hopCount_array1)
sorted_data2, cdf_data2 = calculate_cdf(hopCount_array2)
sorted_data3, cdf_data3 = calculate_cdf(hopCount_array3)
sorted_data4, cdf_data4 = calculate_cdf(hopCount_array4)
sorted_data5, cdf_data5 = calculate_cdf(hopCount_array5)

sorted_data6, cdf_data6 = calculate_cdf(hopCount_array6)
sorted_data7, cdf_data7 = calculate_cdf(hopCount_array7)

# Plot the CDF for each column

labels = []
lines = []
for i in range(0,len(cdf_data6)):
    plt.step(sorted_data6[i], cdf_data6[i], linestyle='-')
lines.append(Line2D([0], [0], linestyle='-'))
#labels.append('1ms')
labels.append('50s')   
for i in range(0,len(cdf_data7)):
    plt.step(sorted_data7[i], cdf_data7[i], linestyle='--')
lines.append(Line2D([0], [1], linestyle='--'))
#labels.append('10ms') 
labels.append('70s')  
for i in range(0,len(cdf_data1)):
    plt.step(sorted_data1[i], cdf_data1[i], linestyle='-')
lines.append(Line2D([0], [0], linestyle='-'))
#labels.append('1ms')
labels.append('90s')   
for i in range(0,len(cdf_data2)):
    plt.step(sorted_data2[i], cdf_data2[i], linestyle='--')
lines.append(Line2D([0], [1], linestyle='--'))
#labels.append('10ms') 
labels.append('110s')   
for i in range(0,len(cdf_data3)):
    plt.step(sorted_data3[i], cdf_data3[i], linestyle='dotted')
lines.append(Line2D([0], [2], linestyle='dotted'))
#labels.append('100ms') 
labels.append('130s')    
for i in range(0,len(cdf_data4)):
    plt.step(sorted_data4[i], cdf_data4[i], linestyle=':')
lines.append(Line2D([0], [3], linestyle=':'))
#labels.append('1000ms') 
labels.append('150s')   
for i in range(0,len(cdf_data5)):
    plt.step(sorted_data5[i], cdf_data5[i], linestyle='-.')
lines.append(Line2D([0], [4], linestyle='-.'))
#labels.append('10000ms') 
labels.append('170s')
#plt.step(sorted_data, cdf_data)
plt.xlabel('Hop Count')
plt.ylabel('CDF')
plt.legend(lines, labels)
plt.grid(True)
# plt.savefig('HopCount_NAC_A2G_scenario_CDF_DSPR.pdf', format='pdf', dpi=1200)
# plt.savefig('HopCount_NAC_A2G_scenario_CDF_GPSR.pdf', format='pdf', dpi=1200)
plt.show()
