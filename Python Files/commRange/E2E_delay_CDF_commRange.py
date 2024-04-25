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


# # # # Find and plot the CDF for each column
for i, range in enumerate(commRange_values):
    group = filtered_e2e_delay_vector[filtered_e2e_delay_vector['commRange'] == range]
    group1 = filtered_e2e_delay_vector_gpsr[filtered_e2e_delay_vector_gpsr['commRange'] == range]
    e2e_delay = group['vecvalue']
    e2e_delay_gpsr = group1['vecvalue']
    sorted_data = np.sort(e2e_delay)
    sorted_data_gpsr = np.sort(e2e_delay_gpsr)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    cdf_gpsr = np.arange(1, len(sorted_data_gpsr) + 1) / len(sorted_data_gpsr)
    # # cdf_df = pd.DataFrame({'vecvalue': sorted_data, 'CDF': cdf})
    linestyle = '--' if i % 2 == 0 else '-'
    # plt.step(sorted_data, cdf, label=f'{range} km', linestyle=linestyle)  ###uncomment for DSPR plot
    # plt.step(sorted_data_gpsr, cdf_gpsr, label=f'{range} km', linestyle=linestyle)   ###uncomment for GPSR plot


plt.xlabel('E2E Delay in seconds')
plt.ylabel('CDF')
plt.legend()
plt.grid(True)
# plt.savefig('E2E_Delay_NAC_A2G_scenario_CommRange_CDF_DSPR.pdf', format='pdf', dpi=1200)
# plt.savefig('E2E_Delay_NAC_A2G_scenario_CommRange_CDF_GPSR.pdf', format='pdf', dpi=1200)
plt.show()