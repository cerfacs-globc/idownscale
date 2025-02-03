import sys
sys.path.append('.')

import glob
import pandas as pd
import matplotlib.pyplot as plt

from iriscc.settings import METRICS_DIR, DATES_TEST, GRAPHS_DIR

exp = str(sys.argv[1])
eval = str(sys.argv[2]) # ex : era5 or y
freq = str(sys.argv[3]) # ex : daily or monthly
test_name = str(sys.argv[4]) # ex : test1,test2,test3
test_list = [str(x) for x in test_name.split(',')]

startdate = DATES_TEST[0].date().strftime('%d/%m/%Y')
enddate = DATES_TEST[-1].date().strftime('%d/%m/%Y')
period = f'{startdate} - {enddate}'

dataframes = []
df_names = ['unet', 'unet_continents', 'unet_none', 'unet_6mb', 'unet_30y', 'swin2sr', 'swin2sr_6mb', 'swin2sr_30y', 'swinunetr', 'swinunetr_6mb', 'swinunetr_6mb_30y']  

for test in test_list:
    if eval == 'y':
        file = glob.glob(str(METRICS_DIR / f'{exp}/mean_metrics/metrics_test_mean_{freq}_{exp}_{test}.csv'))[0]
    else : 
        print(test)
        file = glob.glob(str(METRICS_DIR / f'{exp}/mean_metrics/metrics_test_mean_{freq}_{eval}_{exp}_{test}.csv'))[0]
    df = pd.read_csv(file, delimiter=',', index_col=0) 
    dataframes.append(df)
    #df_names.append(test) 

columns = dataframes[0].columns  
metrics_dict = {}
for col in columns:
    new_df = pd.DataFrame(
        {name: df[col] for name, df in zip(df_names, dataframes)}
    ).T
    new_df.columns = dataframes[0].index 
    metrics_dict[col] = new_df


for key, df in metrics_dict.items():
    ax = df.plot.bar(rot=0, figsize=(18,4))
    plt.axhline(y=0)
    plt.title(f'{key} {eval} ({period})')
    plt.legend(loc='lower right')
    if key.startswith(('rmse', 'bias')):
        plt.ylabel('K')
    if key.startswith('corr'):
        plt.ylim(0.7,1)
    plt.savefig(f"{GRAPHS_DIR}/metrics/{exp}/{key}_barplot_{freq}_{eval}.png") 
