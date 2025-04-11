import sys
sys.path.append('.')

import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from iriscc.settings import METRICS_DIR, DATES_TEST, GRAPHS_DIR

exp = str(sys.argv[1])
target = str(sys.argv[2]) # ex : safran, eobs
test_name = str(sys.argv[3]) # ex : test1,test2,test3
test_list = [str(x) for x in test_name.split(',')]
scale = str(sys.argv[4]) # {scale}, {scale}

startdate = DATES_TEST[0].date().strftime('%d/%m/%Y')
enddate = DATES_TEST[-1].date().strftime('%d/%m/%Y')
period = f'{startdate} - {enddate}'

list_data_mean = []
list_data = []
#df_names = []
test_names = ['CNRM-CM6-1', 'CNRM-CM6-1 bc']
df_names = test_names

for test in test_list:
    print(test)
    file_mean = METRICS_DIR / f'{exp}/mean_metrics/metrics_test_mean_{scale}_{exp}_{test}.csv'
    file = METRICS_DIR / f'{exp}/mean_metrics/metrics_test_{scale}_{exp}_{test}.npz'
    data = dict(np.load(file, allow_pickle=True))
    list_data.append(data)
 
    df_mean = pd.read_csv(file_mean, delimiter=',', index_col=0) 
    list_data_mean.append(df_mean)
    #df_names.append(test) 



metrics_dict_mean = {}
for col in list_data_mean[0].columns:
    new_df_mean = pd.DataFrame(
        {name: df[col] for name, df in zip(test_names, list_data_mean)}
    ).T
    new_df_mean.columns = list_data_mean[0].index 
    metrics_dict_mean[col] = new_df_mean


metrics_dict = {}
for col in list_data[0].keys():
    new_df = pd.DataFrame(
        {name: dict[col] for name, dict in zip(df_names, list_data)}
    )
    metrics_dict[col] = new_df


for key, df in metrics_dict_mean.items():
    ax = df.plot.bar(rot=0, figsize=(6,4))
    plt.axhline(y=0)
    plt.title(f'{key} ({period})')
    plt.legend(loc='lower right')
    if key.startswith(('rmse', 'bias')):
        plt.ylabel('K')
    if key.startswith('corr'):
        plt.ylim(0.9,1)
    plt.savefig(f"{GRAPHS_DIR}/metrics/{exp}/{key}_barplot_{scale}_{target}.png") 


fig, axes = plt.subplots(2, 2, figsize=(8, 6))  
fig.suptitle(f'{scale}-{period} {target} test dataset (swinunet)')
axes = axes.flatten() 
lim = np.array([[0, 7], [-3,3], [0, 1], [0.4, 1]])
for i, (key, df) in enumerate(metrics_dict.items()):
    sns.boxplot(data=df, ax=axes[i], width=0.5, palette="Set2")
    axes[i].set_title(f"{key}")
    #axes[i].set_ylim(lim[i,0], lim[i,1])
    if key.startswith(('rmse', 'bias')):
        axes[i].set_ylabel("K")
        

plt.tight_layout(h_pad=3, w_pad=3)
plt.savefig(f"{GRAPHS_DIR}/metrics/{exp}/boxplot_{scale}_{target}.png") 
