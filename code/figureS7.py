#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
#rc('text', usetex=False)
mpl.rcParams['mathtext.default'] = 'bf'


#%%
### DATA IMPORT ###

# ADCC data using CD56+ cells
file_cd56 = '../data/metaanalysis_tx007.csv'
data_cd56 = pd.read_csv(
        file_cd56,
        comment='#',
        index_col=[0],
        dtype={'donor': str},
)

# ADCC data using PBMCs
# in this figure only EC50 and E:T are required
file_pbmc = '../data/metaanalysis_tx006.csv'
data_pbmc = pd.read_csv(
        file_pbmc,
        comment='#',
        usecols=['donor', 'E:T', 'EC50'],
        dtype={'donor': str},
)


#%%
### DATA PREPARATION ###

# nothing special required, only extract some arrays for ADCC using CD56+ cells
bottom = data_cd56['bottom']
top = data_cd56['top']

# and a colormap (plt.plot has no cmap arg)
colors_ = plt.cm.nipy_spectral(np.linspace(0,1,30))

# and finally a colormap for each histrogram
colors_hist = {
        '30:1': 'brown',
        '15:1': 'orange',
        '6:1' : 'mediumseagreen',
        '3:1' : 'cornflowerblue',
}


#%%
### FIGURE ###

fig = plt.figure(figsize = (8,7))

# we loop over the 4 different E:T ratio and plot the corresponding histogram
for ratio in data_pbmc['E:T'].unique():
        sns.kdeplot(
                data_pbmc[data_pbmc['E:T'] == ratio]['EC50'],
                shade=False,
                color=colors_hist[ratio],
                linewidth=4,
                legend=False,
        )
        
        sns.rugplot(
                data_pbmc[data_pbmc['E:T'] == ratio]['EC50'],
                color=colors_hist[ratio]
        )

plt.ylabel(
        "density",
         fontdict={'size': 10, 'weight': 'bold'})
plt.yticks(size=11)
plt.xlabel(r"log$_{10}$EC$_{50}$ (µg/mL)",
           fontdict={'size': 10, 'weight': 'bold'})
plt.axvline(
        x=data_cd56['EC50'].mean(),
        ls='--',
        linewidth=4,
        color='red',
)
plt.text(
        x=-2.9,
        y=.78,
        s='mean value for isolated CD56+ cells',
        fontdict={
                'size':10,
                'color':'red'
        },
)
plt.ylim((0, 0.82))
plt.xlim((-4.5, 0))
plt.xticks(size=11)
plt.title(
        r'Improved ADCC function avidity upon CD56$^+$ isolation',
        fontdict=dict(size=12, weight='bold')
)


# remake the legend to include the results of the paired t-tests
leg = [
        '30:1 (**)',
       '15:1 (***)',
       '6:1 (***)',
       '3:1 (****)'
        ]

plt.legend(
        leg,
        loc='upper left',
        fontsize=10,
        title='PBMCs (E:T)',
        title_fontsize=12,
)


# a few additional aesthetic
plt.tight_layout()
sns.despine()
plt.savefig('../figures/figureS7.svg') # save as .svg via Visual Code Plots Explorer


#####
# %%
