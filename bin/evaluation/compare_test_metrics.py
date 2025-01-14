import sys
sys.path.append('.')

import glob
import pandas as pd
import matplotlib.pyplot as plt

from iriscc.settings import METRICS_DIR, DATES_TEST, GRAPHS_DIR

exp = str(sys.argv[1])
test_name = str(sys.argv[2]) # ex : test1,test2,test3
test_list = [str(x) for x in test_name.split(',')]

startdate = DATES_TEST[0].date().strftime('%d/%m/%Y')
enddate = DATES_TEST[-1].date().strftime('%d/%m/%Y')
period = f'{startdate} - {enddate}'

dataframes = []
df_names = []  

for test in test_list:
    file = glob.glob(str(METRICS_DIR / f'{exp}/mean_metrics/metrics_test_mean_{exp}_{test}.csv'))[0]
    df = pd.read_csv(file, delimiter=',', index_col=0) 
    dataframes.append(df)
    df_names.append(test) 

columns = dataframes[0].columns  
metrics_dict = {}
for col in columns:
    new_df = pd.DataFrame(
        {name: df[col] for name, df in zip(df_names, dataframes)}
    ).T
    new_df.columns = dataframes[0].index 
    metrics_dict[col] = new_df

for key, df in metrics_dict.items():
    plt.figure(figsize=(12, 6))
    ax = df.plot.bar(rot=0)
    plt.axhline(y=0)
    plt.title(f'{key} ({period})')
    plt.savefig(f"{GRAPHS_DIR}/metrics/{exp}/{key}_barplot.png") 
