##### FIGURE SUPP 5 - PCA PBMCs

# Figure A - PCA using FCGR panel
# Figure B - PCA using NCR/KIR/KLR panel
# Figure C - PCA using immune subset panel


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import matplotlib
#rc('text', usetex=False)
matplotlib.rcParams['mathtext.default'] = 'bf'

from functools import reduce # for merging multiple dataframes

from sklearn.decomposition import PCA


#%%
### DATA IMPORT ###

#  raw data from the ADCC experient tx006
file_tx = '../data/metaanalysis_tx006.csv'
data_tx = pd.read_csv(file_tx,
                      comment='#',
                      index_col=[0],
                      dtype={'donor': str})

# results of the genotyping for all donors
file_allo = '../data/allotypes_pbmcs.csv'
allotype = pd.read_csv(file_allo,
                       comment='#',
                       usecols=['donor', 'FCGR3A'],
                       dtype={'donor': str})
# we don't set yet donor as index because the feature will be required for ANOVA

# flow cytometry raw data for heatmap and correlation analysis
file_fc = '../data/data_all_fc021.csv'
data_fc = pd.read_csv(file_fc,
                      comment='#',
                      dtype={'donor': str})


# we include the %max ADCC from metaanalysis into data_fc -> data
data_top  = data_tx[(data_tx['E:T']=='30:1')][['donor', 'top']]

# and finally merge flow cytometry, ADCC and haplotype for each donor
data = reduce(lambda df1, df2: pd.merge(df1, df2, on='donor'),
              [data_fc, data_top, allotype])

data.rename(columns={'top': 'max % ADCC'}, inplace=True) # good for the legend title

data.set_index('donor', inplace=True, drop=True)


#%%
### PREPARATION OF THE PCA DATA POINTS ###

### PCA for panel A
feat_a = ['nk', 'cd56hicd16-', 'cd56intcd16hi', 'cd56locd16-',
          'nk_cd16b+', 'nk_cd32+', 'nk_cd64+',
          'nk_fcrl3+', 'nk_fcrl5+', 'nk_fcrl6+',
          'lin+cd16+', 'lin+cd16b+', 'lin+cd32+', 'lin+cd64+',
          'lin+fcrl3+', 'lin+fcrl5+', 'lin+fcrl6+']
X_a = data[feat_a]

pca_a = PCA(n_components=2)
comp_a = pca_a.fit_transform(X_a)

pca_data_a = pd.DataFrame(data=comp_a,
                          columns=['PC1', 'PC2'],
                          index=X_a.index)
pca_data_a = pd.merge(pca_data_a, data, left_index=True, right_index=True) # merge on both indices


### PCA for panel B
feat_b = ['pd1+nk', 'cd57+nk',
          'nkp30+nk', 'nkp44+nk', 'nkp46+nk',
          'nkg2a+nk', 'nkg2c+nk', 'nkg2d+nk',
          'kir2dl1+nk', 'kir2dl2+nk', 'kir3dl1+nk',
          'pd1+nkt', 'cd57+nkt',
          'nkp30+nkt', 'nkp44+nkt', 'nkp46+nkt',
          'nkg2a+nkt', 'nkg2c+nkt', 'nkg2d+nkt',
          'kir2dl1+nkt', 'kir2dl2+nkt', 'kir3dl1+nkt']
X_b = data[feat_b]

pca_b = PCA(n_components=2)
comp_b = pca_b.fit_transform(X_b)

pca_data_b = pd.DataFrame(data=comp_b,
                          columns=['PC1', 'PC2'],
                          index=X_b.index)
pca_data_b = pd.merge(pca_data_b, data, left_index=True, right_index=True) # merge on both indices


### PCA for panel C
feat_c = ['t_cell', 'b_cell', 'dc', 'mono', 'gr_nk', 'cd4', 'cd8',
          'tcrgd', 'nkt(cd56)', 'nkt_cd16+', 'tcrgd_cd16+',
          'cd8_cm', 'cd8_e', 'cd8_em', 'cd8_n',
          'cd4_cm', 'cd4_e', 'cd4_em', 'cd4_n', 'cd4_activ', 'cd4_reg',
          'mono_classic', 'mono_non-classic', 'mono_intermed', 'mdc', 'pdc',
          'baso', 'granulo', 'nk']
X_c = data[feat_c]

pca_c = PCA(n_components=2)
comp_c = pca_c.fit_transform(X_c)

pca_data_c = pd.DataFrame(data=comp_c,
                          columns=['PC1', 'PC2'],
                          index=X_c.index)
pca_data_c = pd.merge(pca_data_c, data, left_index=True, right_index=True) # merge on both indices


#%%
### FIGURE ###

fig = plt.figure(figsize = (8,7))

### subplot A - FCGRS
ax1 = plt.subplot(221) 

sns.scatterplot(pca_data_a.loc[:, 'PC1'], pca_data_a.loc[:, 'PC2'],
                alpha=.7,
                size=pca_data_a['max % ADCC'],
                sizes=(10, 200),
                hue=pca_data_a['FCGR3A'],
                ax=ax1)

for donor in pca_data_a.index:
    plt.annotate(donor, (pca_data_a.loc[donor, 'PC1'],
                         pca_data_a.loc[donor, 'PC2']),
                fontsize=8)

plt.title("FcγRs and FcRLs",
          fontdict=dict(size=8, weight='bold'))
plt.xlabel(f"Principal Component 1 ({100*pca_a.explained_variance_ratio_[0]:.1f}%)",
           fontdict={'size': 8, 'weight': 'bold'})
plt.ylabel(f"Principal Component 2 ({100*pca_a.explained_variance_ratio_[1]:.1f}%)",
           fontdict={'size': 8, 'weight': 'bold'})
plt.xticks(size=6)
plt.yticks(size=6)

ax1.get_legend().remove()


### subplot B - NCR/KIR/KLR
ax2 = plt.subplot(222) 

sns.scatterplot(pca_data_b.loc[:, 'PC1'], pca_data_b.loc[:, 'PC2'],
                alpha=.7,
                size=pca_data_b['max % ADCC'],
                sizes=(10, 200),
                hue=pca_data_b['FCGR3A'],
                ax=ax2)

for donor in pca_data_b.index:
    plt.annotate(donor, (pca_data_b.loc[donor, 'PC1'],
                         pca_data_b.loc[donor, 'PC2']),
                 fontsize=8)

plt.title("NCRs/KIRs/KLRs",
          fontdict=dict(size=8, weight='bold'))
plt.xlabel(f"Principal Component 1 ({100*pca_b.explained_variance_ratio_[0]:.1f}%)",
           fontdict={'size': 8, 'weight': 'bold'})
plt.ylabel(f"Principal Component 2 ({100*pca_b.explained_variance_ratio_[1]:.1f}%)",
           fontdict={'size': 8, 'weight': 'bold'})
plt.xticks(size=6)
plt.yticks(size=6)

ax2.get_legend().remove()


### subplot C - immune subsets
plt.subplot(223) 

sns.scatterplot(pca_data_c.loc[:, 'PC1'], pca_data_c.loc[:, 'PC2'],
                alpha=.7,
                size=pca_data_c['max % ADCC'],
                sizes=(10, 200),
                hue=pca_data_c['FCGR3A'])

for donor in pca_data_c.index:
    plt.annotate(donor, (pca_data_c.loc[donor, 'PC1'],
                         pca_data_c.loc[donor, 'PC2']),
                fontsize=8)

plt.title("immune populations",
          fontdict=dict(size=8, weight='bold'))
plt.xlabel(f"Principal Component 1 ({100*pca_c.explained_variance_ratio_[0]:.1f}%)",
           fontdict={'size': 8, 'weight': 'bold'})
plt.ylabel(f"Principal Component 2 ({100*pca_c.explained_variance_ratio_[1]:.1f}%)",
           fontdict={'size': 8, 'weight': 'bold'})
plt.xticks(size=6)
plt.yticks(size=6)


plt.tight_layout() # first tight then add the legend on the remaining space


plt.legend(bbox_to_anchor=(1.25, .92),
           loc='upper left',
           fontsize=8,
           title=None)


# a few additional aesthetic
fig.text(0.01, 0.965, "A", weight="bold", size=16, horizontalalignment='left')
fig.text(0.50, 0.965, "B", weight="bold", size=16, horizontalalignment='left')
fig.text(0.01, 0.475, "C", weight="bold", size=16, horizontalalignment='left')

#plt.savefig('../figures/figureS5.svg')
plt.savefig('../figures/figureS5.pdf')


#####