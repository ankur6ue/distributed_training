import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# colnames=['TIME', 'b=1']
# df_1 = pd.read_csv(os.path.join('logs_test/finetune/gpt2', 'b_1.csv'), names=colnames, header=None).set_index('TIME')
colnames=['TIME', 'b=24']
df_2 = pd.read_csv(os.path.join('results/bert_small', '20-03-07.csv'), names=colnames, header=None).set_index('TIME')
# colnames=['TIME', 'b=3']
# df_3 = pd.read_csv(os.path.join('logs_test/finetune/gpt2', 'b_3.csv'), names=colnames, header=None).set_index('TIME')
# colnames=['TIME', 'b=4']
# df_4 = pd.read_csv (os.path.join('logs_test/finetune/gpt2', 'b_4.csv'), names=colnames, header=None).set_index('TIME')
# colnames=['TIME', 'b=3 (ac)']
# df_4 = pd.read_csv (os.path.join('logs_test/finetune/gpt2', 'b_3_ac.csv'), names=colnames, header=None).set_index('TIME')

colnames=['TIME', 'b=24 (ddp)']
df_5 = pd.read_csv (os.path.join('results/bert_small', '12-56-42.csv'), names=colnames, header=None).set_index('TIME')

df = [df_2, df_5]

min = []
max = []
for df_ in df:
    min.append(np.floor(df_.index.min()))
    max.append(np.floor(df_.index.max()))

min_ = np.max(min)
max_ = np.min(max)
idx = np.arange(min_, max_, 0.5)

cnt = 0
for df_ in df:
    df_ = df_.reindex(df_.index.union(idx)).interpolate(method='linear').reindex(idx)
    df[cnt] = df_
    cnt = cnt + 1


d_merged = pd.concat(df, axis=1, join="inner")

ax = d_merged.plot()
ax.set_ylabel('Percent Complete')
plt.show()

