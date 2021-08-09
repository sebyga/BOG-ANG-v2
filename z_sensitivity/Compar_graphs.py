# Compare_graphs.py
# %% Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% Storage data
m100 = [0.609656	,0.577206,	0.564785	,	0.555954, 0.546625]
m300 = [1.384767	,1.294327,	1.257901	,	1.231283, 1.202443]
# %% Temperature data
T100 = [74.0299	, 83.576234,	87.494439	, 90.377959,	93.51689]
T300 = [80.096485, 94.025188	,	100.159211	, 104.857113,	110.169397]
# %% Bar Graph (Storage)
xaxlab = ['z=0.5','z=0.8','z=1.0','z=1.2','z=1.5']
plt.figure(dpi = 85)
xax = np.array([1,2,3,4,5])
plt.bar(xax-0.2, m100,color = [0.3,0.7,0.3],
label = '100 bar',width = 0.4)  
plt.bar(xax+0.2, m300,color = [0.2,0.5,0.2],
label = '300 bar',width = 0.4)

plt.ylabel('Gas storage (kg/kg)',fontsize = 12.5)
plt.ylim([0,2.0])
plt.xticks(xax,labels = xaxlab,fontsize = 13.5)
plt.legend(fontsize = 13)
plt.savefig('storage_compare.png')

# %% Bar Graph (Temperature)
plt.figure(dpi = 85)
xax = np.array([1,2,3,4,5])
plt.bar(xax-0.2, np.array(T100)+25,color = 'orange',
label = '100 bar',width = 0.4)
plt.bar(xax+0.2, np.array(T300)+25,color = 'r',
label = '300 bar', width = 0.4)

plt.ylabel(r'Temperature ($^{o}$C)',fontsize = 12.5)
plt.ylim([0,220])
plt.xticks(xax,labels = xaxlab,fontsize = 13.5)
plt.legend(fontsize = 13)
plt.savefig('temperature_compare.png')
# %%
