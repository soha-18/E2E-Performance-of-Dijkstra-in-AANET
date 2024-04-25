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
    df['run'] = df.index
    df['commRange'] = df['run'].apply(calculate_comm_range)
    df = df[['commRange'] + [col for col in df.columns if col != 'commRange']]
    # df = df.drop(columns=['vectime'])
    df = df.rename(columns={'run': 'SimulationRun'})
    return df

def process_dataframe_gpsr(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
    df.reset_index(drop=True, inplace=True)
    df['commRange'] = df['run'].apply(calculate_comm_range)
    df = df[['commRange'] + [col for col in df.columns if col != 'commRange']]
    # df = df.drop(columns=['vectime'])
    df = df.rename(columns={'run': 'SimulationRun'})
    return df

def process_dataframe_gpsr1(df):
    df['run'] = df['run'].str.extract(r'-(\d+)-')
    df = df[df['type'] != 'attr']
    df = df.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
    df.reset_index(drop=True, inplace=True)
    df['commRange'] = df['run'].apply(calculate_comm_range1)
    df = df[['commRange'] + [col for col in df.columns if col != 'commRange']]
    # df = df.drop(columns=['vectime'])
    df = df.rename(columns={'run': 'SimulationRun'})
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
       

file_path = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Comm_Range/e2e_delay_vector.csv'
file_path_gpsr = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/e2e_delay_vector.csv'
file_path1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/DSPR/Comm_Range/e2e_delay_vector1.csv'
file_path_gpsr1 = '~/ma_sohini_maji/Master_Thesis/Results/CSV Files/GPSR/Comm_Range/e2e_delay_vector1.csv'

df = pd.read_csv(file_path)
df1 = pd.read_csv(file_path_gpsr)
df2 = pd.read_csv(file_path1)
df3 = pd.read_csv(file_path_gpsr1)

column_to_check = 'name'
df = df.dropna(subset=[column_to_check])
df1 = df1.dropna(subset=[column_to_check])
df2 = df2.dropna(subset=[column_to_check])
df3 = df3.dropna(subset=[column_to_check])


df = process_dataframe(df)
df['vecvalue'] = df['vecvalue'].str.split()
df['vectime'] = df['vectime'].str.split()


df1 = process_dataframe_gpsr(df1)
df1['vecvalue'] = df1['vecvalue'].str.split()
df1['vectime'] = df1['vectime'].str.split()

df2['run'] = df2['run'].str.extract(r'-(\d+)-')
df2 = df2[df2['type'] != 'attr']
df2 = df2.drop(columns=['name','type','module', 'attrname', 'attrvalue', 'value'])
# df2 = df2.drop(columns=['vectime'])
df2['commRange'] = 370
df2 = df2[['commRange'] + [col for col in df2.columns if col != 'commRange']]
df2['run'] = df2['run'].astype(int)
df2.loc[(df2['run'] >= 0) & (df2['run'] <= 9), 'run'] += 40
df2 = df2.rename(columns={'run': 'SimulationRun'})
df2['vectime'] = df2['vectime'].str.split()
df2['vecvalue'] = df2['vecvalue'].str.split()

df3 = process_dataframe_gpsr1(df3)
df3['SimulationRun'] = df3['SimulationRun'].astype(int)
df3.loc[(df3['SimulationRun'] >= 0) & (df3['SimulationRun'] <= 19), 'SimulationRun'] += 30
df3['vectime'] = df3['vectime'].str.split()
df3['vecvalue'] = df3['vecvalue'].str.split()

e2e_delay_vector = pd.concat([df, df2])
e2e_delay_vector_gpsr = pd.concat([df1, df3])

e2e_delay_vector = pd.DataFrame({
    'commRange': e2e_delay_vector['commRange'].repeat(e2e_delay_vector['vecvalue'].apply(len)),
    'SimulationRun': e2e_delay_vector['SimulationRun'].repeat(e2e_delay_vector['vecvalue'].apply(len)),
    'vectime' : [x for sublist in e2e_delay_vector['vectime'] for x in sublist],
    'vecvalue': [x for sublist in e2e_delay_vector['vecvalue'] for x in sublist]
})
e2e_delay_vector['vectime'] = pd.to_numeric(e2e_delay_vector['vectime'], errors='coerce')
e2e_delay_vector['vecvalue'] = pd.to_numeric(e2e_delay_vector['vecvalue'], errors='coerce')
filtered_e2e_delay_vector = e2e_delay_vector[(e2e_delay_vector['vectime'] >= 200.0) & (e2e_delay_vector['vectime'] <= 1500.0)]
filtered_e2e_delay_vector = filtered_e2e_delay_vector.drop(columns=['vectime'])
filtered_e2e_delay_vector.reset_index(drop=True, inplace=True)
# print(e2e_delay_vector)

e2e_delay_vector_gpsr = pd.DataFrame({
    'commRange': e2e_delay_vector_gpsr['commRange'].repeat(e2e_delay_vector_gpsr['vecvalue'].apply(len)),
    'SimulationRun': e2e_delay_vector_gpsr['SimulationRun'].repeat(e2e_delay_vector_gpsr['vecvalue'].apply(len)),
    'vectime' : [x for sublist in e2e_delay_vector_gpsr['vectime'] for x in sublist],
    'vecvalue': [x for sublist in e2e_delay_vector_gpsr['vecvalue'] for x in sublist]
})

e2e_delay_vector_gpsr['vectime'] = pd.to_numeric(e2e_delay_vector_gpsr['vectime'], errors='coerce')
e2e_delay_vector_gpsr['vecvalue'] = pd.to_numeric(e2e_delay_vector_gpsr['vecvalue'], errors='coerce')
filtered_e2e_delay_vector_gpsr = e2e_delay_vector_gpsr[(e2e_delay_vector_gpsr['vectime'] >= 200.0) & (e2e_delay_vector_gpsr['vectime'] <= 1500.0)]
filtered_e2e_delay_vector_gpsr = filtered_e2e_delay_vector_gpsr.drop(columns=['vectime'])
filtered_e2e_delay_vector_gpsr.reset_index(drop=True, inplace=True)


commRange_values = e2e_delay_vector['commRange'].unique()

Communication_range = ['170', '220', '270', '320', '370']
filtered_e2e_delay_vector = filtered_e2e_delay_vector.groupby(['commRange', 'SimulationRun'])['vecvalue'].mean().reset_index()
filtered_e2e_delay_vector_gpsr = filtered_e2e_delay_vector_gpsr.groupby(['commRange', 'SimulationRun'])['vecvalue'].mean().reset_index()
sample_mean_E2E_delay = []
margin_of_error_array = []
sample_mean_E2E_delay_gpsr = []
margin_of_error_array_gpsr = []

for commRange in commRange_values:
    group1 = filtered_e2e_delay_vector[filtered_e2e_delay_vector['commRange'] == commRange]
    group2 = filtered_e2e_delay_vector_gpsr[filtered_e2e_delay_vector_gpsr['commRange'] == commRange]
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
    
# # CI Plot
plt.errorbar(x = Communication_range, y= sample_mean_E2E_delay, yerr=margin_of_error_array, fmt='D',capsize=4, capthick=1, color='m', markersize=8,label='DSPR')
plt.errorbar(x = Communication_range, y= sample_mean_E2E_delay_gpsr, yerr=margin_of_error_array_gpsr, fmt='o',capsize=4, capthick=1, color='b', markersize=8,label='GPSR')
plt.xlabel('Communication Range in kilometres')
plt.ylabel('E2E Delay in seconds')
plt.legend()
plt.grid(True)
# plt.savefig('E2E Delay_NAC_A2G_scenario_commRange_CI.pdf', format='pdf', dpi=1200)
plt.show()
