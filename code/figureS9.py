##### FIGURE SUPP 9 - PCAs CD56 SUBSETS

# Figure A - PCA FCGRS NK
# Figure B - PCA FCGRS CD56+lin+
# Figure C - PCA FCGRS TCRgd
# Figure D - PCA NCR NK
# Figure E - PCA KIR NK
# Figure F - PCA KLR NK
# Figure G - PCA NCR CD56+lin+
# Figure H - PCA KIR CD56+lin+
# Figure I - PCA KLR CD56+lin+


#%%
import numpy as np
import pandas as pd

from functools import reduce # for merging multiple dataframes

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.style.use('default')
#rc('text', usetex=False)
mpl.rcParams['mathtext.default'] = 'bf' # use greek chars from the clipboard

#from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#%%
### DATA IMPORT ###

# flow cytometry data from experiment fc022 gated on CD56+
file_fc = '../data/data_all_fc022.csv'
data_fc = pd.read_csv(file_fc,
                        comment="#",
                        dtype={'donor': str})
data_fc.set_index('donor', drop=True, inplace=True)

# ADCC data from experiment tx007 using isolated CD56+ cells
file_adcc = '../data/metaanalysis_tx007.csv'
data_adcc = pd.read_csv(file_adcc,
                        comment='#',
                        index_col=[0],
                        dtype={'donor': str})

# results of the genotyping for all donors
file_allo = '../data/allotypes_pbmcs.csv'
allotype = pd.read_csv(file_allo,
                       comment='#',
                       usecols=['donor', 'FCGR3A'],
                       dtype={'donor': str})


#%%
### PREPARATION OF THE DATA SET ###

# for the PCA analysis and bubble plot we extract the %max ADCC from metaanalysis
data_top  = data_adcc[['donor', 'top']]

# and we merge flow cytometry, ADCC and haplotype data together -> data
data = reduce(lambda df1, df2: pd.merge(df1, df2, on='donor'),
              [data_fc, data_top, allotype])
data.rename(columns={'top': 'max % ADCC'}, inplace=True) # best for the legend title
data.set_index('donor', inplace=True, drop=True)

# selection of the features for each individual PCA
# FCGR for NK
feat_nk_fcgr = ['nk_cd56int|cd16hi', 'nk_cd16b+', 'nk_cd32+', 'nk_cd64+',
                'nk_fcrl3+', 'nk_fcrl5+', 'nk_fcrl6+']
       
# FCGR for 'NKT' i.e. CD56+lin+
feat_nkt_fcgr = ['nkt_cd16+', 'nkt_cd16b+', 'nkt_cd32+', 'nkt_cd64+',
                 'nkt_fcrl3+', 'nkt_fcrl5+', 'nkt_fcrl6+']

# FCGR for TCRgd
feat_tcrgd_fcgr = ['tcrgd_cd16+', 'tcrgd_cd16b+', 'tcrgd_cd32+', 'tcrgd_cd64+',
                   'tcrgd_fcrl3+', 'tcrgd_fcrl5+', 'tcrgd_fcrl6+']

# NCR for NK
feat_nk_ncr = ['nkp30+|nk', 'nkp44+|nk', 'nkp46+|nk']

# KLR for NK
feat_nk_klr = ['nkg2a+|nk', 'nkg2c+|nk', 'nkg2d+|nk']

# KIR for NK
feat_nk_kir = ['kir2dl1+|nk', 'kir2dl2+|nk', 'kir3dl1+|nk']

# NCR for 'NKT' i.e. CD56+lin+
feat_nkt_ncr = ['nkp30+|nkt', 'nkp44+|nkt', 'nkp46+|nkt']

# KLR for 'NKT' i.e. CD56+lin+
feat_nkt_klr = ['nkg2a+|nkt', 'nkg2c+|nkt','nkg2d+|nkt']

# KIR for 'NKT' i.e. CD56+lin+
feat_nkt_kir = ['kir2dl1+|nkt', 'kir2dl2+|nkt', 'kir3dl1+|nkt']

# we also need a dict for giving a title to the different plots
feat_dict = {
    r'FcγRs/FcRLs in NK cells':           feat_nk_fcgr,
    r'FcγRs/FcRLs in CD56$^+$lineage$^+$':feat_nkt_fcgr,
    r'FcγRs/FcRLs in TCRγδ':              feat_tcrgd_fcgr,
    r'NCRs in NK cells':                  feat_nk_ncr,
    r'KLRs in NK cells':                  feat_nk_klr,
    r'KIRs in NK cells':                  feat_nk_kir,
    r'NCRs in CD56$^+$lineage$^+$':       feat_nkt_ncr,
    r'KLRs in CD56$^+$lineage$^+$':       feat_nkt_klr,
    r'KIRs in CD56$^+$lineage$^+$':       feat_nkt_kir}

# and finally setting the PCA object common for the different analyses
pca = PCA(n_components=2)


#%%
### FIGURE ###
fig = plt.figure(figsize = (8,8))

# For this figure we loop over feat_dict, run the PCA and plot afterwards

i=1 # counter for the subplots

for title, feats in feat_dict.items():
    ax = plt.subplot(3,3,i)
    
    X = data[feats]
    comp = pca.fit_transform(X)

    pca_data = pd.DataFrame(data=comp,
                            columns=['PC1', 'PC2'],
                            index=X.index)
    pca_data.index = data.index
    pca_data = pd.merge(pca_data, data, left_index=True, right_index=True) # merge on both indices

    sns.scatterplot(pca_data.loc[:, 'PC1'], pca_data.loc[:, 'PC2'],
                    alpha=.7,
                    size=pca_data['max % ADCC'],
                    sizes=(5, 150),
                    hue=pca_data['FCGR3A'],
                    ax=ax)

    for donor in pca_data.index:
        plt.annotate(donor, (pca_data.loc[donor, 'PC1'],
                             pca_data.loc[donor, 'PC2']),
                     fontsize=6)
    
    plt.xlabel(f"Principal Component 1 ({100*pca.explained_variance_ratio_[0]:.1f}%)",
               fontdict={'size': 8, 'weight': 'bold'})
    plt.ylabel(f"Principal Component 2 ({100*pca.explained_variance_ratio_[1]:.1f}%)",
               fontdict={'size': 8, 'weight': 'bold'})
    plt.xticks(size=6)
    plt.yticks(size=6)

    plt.title(title,
              fontdict=dict(size=8, weight='bold'))

    ax.get_legend().remove()
    i+=1


# a few additional aesthetic
plt.tight_layout()

plt.legend(loc='lower right',
           fontsize=6,
           title=None)

fig.text(0.010, 0.970, "A", weight="bold", size=16, horizontalalignment='left')
fig.text(0.335, 0.970, "B", weight="bold", size=16, horizontalalignment='left')
fig.text(0.665, 0.970, "C", weight="bold", size=16, horizontalalignment='left')
fig.text(0.010, 0.645, "D", weight="bold", size=16, horizontalalignment='left')
fig.text(0.335, 0.645, "E", weight="bold", size=16, horizontalalignment='left')
fig.text(0.665, 0.645, "F", weight="bold", size=16, horizontalalignment='left')
fig.text(0.010, 0.315, "G", weight="bold", size=16, horizontalalignment='left')
fig.text(0.335, 0.315, "H", weight="bold", size=16, horizontalalignment='left')
fig.text(0.667, 0.315, "I", weight="bold", size=16, horizontalalignment='left')


#plt.savefig('../figures/figureS9.svg')
plt.savefig('../figures/figureS9.pdf')


#####