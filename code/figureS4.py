##### FIGURE SUPP 4 - ADCC PBMCs

# Figure A - Sumarry of the ADCC max (upper asymptote) % cytotoxicity values using
#            whole PBMCs at different E:T ratio and for each individual donor
# Figure B - Sumarry of the ADCC min (lower asymptote) % cytotoxicity values using 
#            whole PBMCs at different E:T ratio and for each individual donor


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'bf' # so that greek car and symbols are bold in plot ticks/labels
mpl.rc('text', usetex=False)

# for inset colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


#%%
### Import and preparation of the data ###

# Raw data from the ADCC experient tx006
file_tx = '../data/metaanalysis_tx006.csv'
data_tx = pd.read_csv(file_tx,
                      comment='#',
                      index_col=[0],
                      dtype={'donor': str})

# Results of the genotyping for all donors
file_allo = '../data/allotypes_pbmcs.csv'
allotype = pd.read_csv(file_allo,
                       comment='#',
                       usecols=['donor', 'FCGR3A'],
                       dtype={'donor': str})

# map the FCGR3A haplotype to each donor in the dataset
data = pd.merge(data_tx, allotype, on='donor', how='left')


#%%
### Building of the 2 parts of the figure ###

fig = plt.figure(figsize=(8,3.7)) # modified A4 format in inches

### Figure A
ax1 = plt.subplot(121)
sns.pointplot(x='E:T', y='top', data=data, 
             hue='donor', 
             order=['3:1', '6:1', '15:1', '30:1'], 
             palette='tab20',
             markers='^',
             ax=ax1)

plt.ylabel(r"maximum % ADCC",
           fontdict={'size': 10, 'weight': 'bold'})
plt.yticks(size=8)
plt.ylim((0,115))
plt.xlabel('E:T ratio',
          fontdict={'size': 10, 'weight': 'bold'})
plt.xticks(rotation=0,
           size=10)

ax1.get_legend().remove() # we use a common legend for the 2 figures (see Figure B)


## Figure B
ax2 = plt.subplot(122)
sns.pointplot(x='E:T', y='bottom', data=data,
             hue='donor',
             order=['3:1', '6:1', '15:1', '30:1'],
             palette='tab20',
             markers='^',
             ax=ax2)

plt.ylabel(r"minimum % ADCC",
           fontdict={'size': 10, 'weight': 'bold'})
plt.yticks(size=8)
plt.ylim((0,115))
plt.xlabel('E:T ratio',
          fontdict={'size': 10, 'weight': 'bold'})
plt.xticks(rotation=0,
           size=10)

plt.legend(ncol=1,
           bbox_to_anchor=(1.05, 1),
           loc='upper left',
           fontsize=7,
           title='donor',
           title_fontsize=8)


# a few additional aesthetic
fig.text(0.01,  0.95, "A", weight="bold", size=16, horizontalalignment='left')
fig.text(0.456, 0.95, "B", weight="bold", size=16, horizontalalignment='left')

plt.tight_layout()

#plt.savefig('../figures/figureS4.svg')
plt.savefig('../figures/figureS4.pdf')


######