##### FIGURE SUPP 7 - ADCC CD56

# Figure A - min/max before/after plot for ADCC with CD56+ for all donors [experiment tx007]
# Figure B - KDE histogram for the EC50 from ADCC with CD56+ cells [experiment tx007]
# Figure C - KDE histograms at different E:T ratios for the EC50 from ADCC with PBMCs [experiment tx006]


#%%
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
from pingouin import ttest

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
#rc('text', usetex=False)
mpl.rcParams['mathtext.default'] = 'bf'


#%%
### DATA IMPORT ###

# ADCC data using CD56+ cells
file_cd56 = '../data/metaanalysis_tx007.csv'
data_cd56 = pd.read_csv(file_cd56,
                        comment='#',
                        index_col=[0],
                        dtype={'donor': str})

# ADCC data using PBMCs
# in this figure only EC50 and E:T are required
file_pbmc = '../data/metaanalysis_tx006.csv'
data_pbmc = pd.read_csv(file_pbmc,
                        comment='#',
                        usecols=['donor', 'E:T', 'EC50'],
                        dtype={'donor': str})


#%%
### DATA PREPARATION ###

# nothing special required, only extract some arrays for ADCC using CD56+ cells
bottom = data_cd56['bottom']
top = data_cd56['top']

# and a colormap (plt.plot has no cmap arg)
colors_ = plt.cm.nipy_spectral(np.linspace(0,1,30))

# and finally a colormap for each histrogram
colors_hist = {'30:1': 'brown',
               '15:1': 'orange',
               '6:1' : 'mediumseagreen',
               '3:1' : 'cornflowerblue'}


#%%
### STAT TESTS ###

f=open("../stats/ttests_figureS7.txt", 'w')

# unpaired t-tests between ALL the 30 log10EC50 values from ADCC using CD56+ cells
# (E:T=5:1) and ALL the __19__ log10EC50 values from ADCC using PBMCs at a different ratios
f.write("Unpaired t-tests between all the 30 EC50 values using isolated CD56+ cells and all the 19 values using PBMCs")
a = data_cd56['EC50']
for ET in data_pbmc['E:T'].unique():
        f.write(f"\nE:T={ET}¬\n")
        b = data_pbmc[data_pbmc['E:T'] == ET]['EC50']
        stats = np.round(ttest_ind(a,b), 6)   # round the results of the t-test for nicer repr
        f.write(f"statistic={stats[0]}, pvalue={stats[1]}")

f.write('\n')
f.write('-'*80)
f.write('\n')

# paired t-test between the 19 log10EC50 values from ADCC using CD56+ cells
# and __19__ log10EC50 values from ADCC using PBMCs at different E:T ratios **from the same donor**
f.write("Paired t-tests between 19 EC50 values using isolated CD56+ cells and 19 values using PBMCs from the corresponding donors")
c = data_cd56.set_index('donor')['EC50']
for ET in data_pbmc['E:T'].unique():
        f.write(f"\nE:T={ET}¬\n")
        d = data_pbmc[data_pbmc['E:T'] == ET].set_index('donor')['EC50']
        temp = pd.merge(c, d, on='donor', suffixes={'_CD56', '_PBMCs'})
        e,d = temp['EC50_CD56'], temp['EC50_PBMCs']
        stats = np.round(ttest_rel(e,d), 6)   # round the results of the t-test for nicer repr
        f.write(f"statistic={stats[0]}, pvalue={stats[1]}")
f.close


#%%
### FIGURE ###

fig = plt.figure(figsize = (8,7))

# ### Subplot A - before/after min-max ADCC using CD56+ cells
# plt.subplot(211)

# # for this before/after graph, we plot bottom-top doublets for each donor
# for x in range(30):
#     plt.plot([0,1], [bottom[x], top[x]],
#              label=data_cd56.iloc[x]['donor'],
#              lw=3,
#              alpha=.5,
#              color=colors_[x],
#              marker='^',
#              markeredgecolor='black',
#              markerfacecolor='k',
#              markersize=8)

# plt.ylabel(r"% specific release",
#            fontdict={'size': 8, 'weight': 'bold'})
# plt.ylim((-10,100))
# plt.yticks(size=6)
# plt.xticks([0,1], ['minimun', 'maximum'],
#            size=9,
#            fontweight= 'bold')

# plt.legend(loc='upper left',
#            fontsize=7,
#            ncol=3,
#            title='donor',
#            title_fontsize=9)


# ### Subplot B - KDE histograms with ADCC data using CD56+ cells
# plt.subplot(223)

# sns.kdeplot(data_cd56['EC50'],
#             bw=.4,
#             shade=True,
#             color='indigo',
#             linewidth=2,
#             legend=False)

# sns.rugplot(data_cd56['EC50'],
#             color='indigo',)

# plt.ylabel("density",
#            fontdict={'size': 8, 'weight': 'bold'})
# plt.yticks(size=6)
# plt.xlabel(r"log$_{10}$EC$_{50}$ (µg/mL)",
#            fontdict={'size': 8, 'weight': 'bold'})
# plt.axvline(x=data_cd56['EC50'].mean(),    # equals -3.0042
#             ls='--',
#             linewidth=2,
#             color='red')
# plt.xlim((-5, 0))
# plt.xticks(size=6)
# plt.title(r'isolated CD56$^+$ cells',
#           fontdict=dict(size=10, weight='bold'))


# ### Subplot C - KDE histograms with ADCC data using PBMCs
ax3 = plt.subplot()

# we loop over the 4 different E:T ratio and plot the corresponding histogram
for ratio in data_pbmc['E:T'].unique():
    sns.kdeplot(data_pbmc[data_pbmc['E:T'] == ratio]['EC50'],
                bw=.4,
                shade=False,
                color=colors_hist[ratio],
                linewidth=4,
                legend=False,
                ax=ax3)

    sns.rugplot(data_pbmc[data_pbmc['E:T'] == ratio]['EC50'],
                color=colors_hist[ratio])

plt.ylabel("density",
           fontdict={'size': 16, 'weight': 'bold'})
plt.yticks(size=12)
plt.xlabel(r"log$_{10}$EC$_{50}$ (µg/mL)",
           fontdict={'size': 16, 'weight': 'bold'})
plt.axvline(x=data_cd56['EC50'].mean(),
            ls='--',
            linewidth=3,
            color='red')
plt.xlim((-5, 0))
plt.xticks(size=12)
# plt.title('whole PBMCs',
#           fontdict=dict(size=10, weight='bold'))


# remake the legend to include the results of the paired t-tests
leg = ['30:1 (**)',
       '15:1 (***)',
       '6:1 (***)',
       '3:1 (****)']

plt.legend(leg,
           loc='upper left',
           fontsize=10,
           title='E:T ratio',
           title_fontsize=12)


# a few additional aesthetic
plt.tight_layout()

# fig.text(0.01, 0.97, "A", weight="bold", size=16, horizontalalignment='left')
# fig.text(0.01, 0.49, "B", weight="bold", size=16, horizontalalignment='left')
# fig.text(0.50, 0.49, "C", weight="bold", size=16, horizontalalignment='left')


plt.savefig('../figures/figureS7.svg') # save as .svg via Visual Code Plots Explorer
#plt.savefig('../figures/figureS7.pdf')


#####