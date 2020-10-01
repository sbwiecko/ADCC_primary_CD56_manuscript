##### FIGURE SUPP 6 - MARKERS CD56+lineage+

# Figure - Expression of markers in CD56+ NKT+ and TCRgd cells


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
#rc('text', usetex=False)
mpl.rcParams['mathtext.default'] = 'rm'


#%%
### DATA IMPORT ###

data = pd.read_csv('../data/data_all_fc022.csv',
                   comment="#",
                   dtype={'donor': str},)
data.set_index('donor', inplace=True, drop=True)


#%%
### DATA PREPARATION ###

# we keep markers of interest for this particular figure
data = data.loc[:,['nkt_cd16b+', 'nkt_cd32+', 'nkt_cd64+',
                   'nkt_fcrl3+', 'nkt_fcrl5+', 'nkt_fcrl6+',
                   'nkp30+|nkt', 'nkp44+|nkt', 'nkp46+|nkt',
                   'nkg2a+|nkt', 'nkg2c+|nkt', 'nkg2d+|nkt',
                   'kir2dl1+|nkt', 'kir2dl2+|nkt', 'kir3dl1+|nkt',
                   'cd57+|nkt', 'pd1+|nkt']]

# mapper for renaming the columns, i.e. cropping and uppercase...
mapper = {feat: feat[4:].upper() for feat in data.columns \
            if feat.startswith('nkt_')}
mapper.update({feat: feat[:-4].upper() for feat in data.columns \
            if feat.endswith('|nkt')})
# ... and some manual renaming
mapper.update({'nkt_cd16b+': 'CD16b+',
               'nkp30+|nkt': 'NKp30+', 'nkp44+|nkt': 'NKp44+', 'nkp46+|nkt': 'NKp46+',
               'nkt_fcrl3+': 'FcRL3+', 'nkt_fcrl5+': 'FcRL5+', 'nkt_fcrl6+': 'FcRL6+',
               'kir2dl1+|nkt': 'KIR2DL1/S1/3/5+', 'kir2dl2+|nkt': 'KIR2DL2/3/S2+',
               'pd1+|nkt': 'PD-1+'})

data = data.rename(columns=mapper)


#%%
### FIGURE ###

fig = plt.figure(figsize = (8,3.5))

ax=sns.stripplot(data=data) # seaborn consideres the data are in the wide format

sns.boxplot(data=data,
            color='white',
            fliersize=0, # there is a stripplot anyway
            ax=ax)

# loop over each box of the axis and attribute black color
for i,artist in enumerate(ax.artists):
    artist.set_edgecolor('black')

# loop over each line of the axis and attribute black color
for i in range(len(ax.lines)):
    line = ax.lines[i]
    line.set_color('black')

plt.ylabel(r'% of total CD56$^+$lineage$^+$ cells',
           fontdict={'size': 8, 'weight': 'bold'})
plt.ylim((0,100))
plt.yticks(size=6)
plt.xticks(rotation=90,
           size=8)


plt.tight_layout()

#plt.savefig('../figures/figureS6.svg')
plt.savefig('../figures/figureS6.pdf')


#####