from robustness_dataset import RobustnessDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

data = RobustnessDataset(path="/home/younan/project_calibration")

# extend tss to full 15625 archi
df = pd.read_csv('./final_results/cifar10_results.csv')

df_extend  =df.copy()
for i in range(15625):
    if data.get_uid(i) != str(i):
        replace_i = int(data.get_uid(i))
        row = df_extend.loc[df_extend['config']==replace_i]
        row.iloc[0, row.columns.get_loc('config')] = i

        df_extend = pd.concat([df_extend, row])

print(len(df))
print(len(df_extend))

df_extend.to_csv('./final_results/cifar10_results_full.csv', index=False)



# extend json like data value to all metrics with corresponding binsize 
df = pd.read_csv('./final_results/cifar10_results.csv')
# extract the value of 'result' corresponding to a specific 'n_bins' or 'num_bins' value
def extract_result(data, target_bins):
    for item in data:
        if ('n_bins' in item and item['n_bins'] == target_bins) or ('num_bins' in item and item['num_bins'] == target_bins):
            return item['result']
    return None

# F\function to apply to each row of the dataframe
def expand_metrics(row):
    # Parse the metrics data using eval()
    for col in ['ece', 'ece_em', 'cwECE', 'cwECE_em']:
        data = eval(row[col])

        # Extract the results for each desired bin size and create new columns
        for bin_size in [5,10,15,20,25,50,100,200,500]:
            col_name = f"{col}_{bin_size}"
            row[col_name] = extract_result(data, bin_size)

    return row

df_extend = df.apply(expand_metrics, axis=1)
df_extend.to_csv('./final_results/cifar10_results_full.csv', index=False)