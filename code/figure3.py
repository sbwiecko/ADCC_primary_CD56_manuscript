#%%
from functools import reduce # for merging multiple dataframes

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.style.use('default')
mpl.rcParams['mathtext.default'] = 'regular' # we keep rm until legend in C the we will use another trick

cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "",
    [
        "cornflowerblue",
        "lightcoral",
        "gainsboro"
    ]
)

from sklearn.decomposition import PCA


#%%
### DATA IMPORT ###

## FC data for total PBMCs as from figure 1
data = pd.read_csv(
    '../data/data_all_fc022.csv',
    comment="#",
    dtype={'donor': str},
)
data.set_index('donor', inplace=True, drop=True)


# flow cytometry data from experiment fc022 gated on CD56+
data_cd56 = pd.read_csv(
    '../data/data_all_fc022_cd56.csv',
    comment="#",
    dtype={'donor': str},
)
data_cd56.set_index('donor', drop=True, inplace=True)

# ADCC data from experiment tx007 using isolated CD56+ cells
data_adcc = pd.read_csv(
    '../data/metaanalysis_tx007.csv',
    comment='#',
    index_col=[0],
    dtype={'donor': str},
)

# results of the genotyping for all donors
allotype = pd.read_csv(
    '../data/allotypes_pbmcs.csv',
    comment='#',
    usecols=['donor', 'FCGR3A'],
    dtype={'donor': str},
)


#%%
### PREPARATION OF THE DATA SETS ###

# we keep the FCGR, FcRL, NCR, KIR and KLR markers for NK cells
list_markers_nk = [
    'nk_cd56int|cd16hi', 'nk_cd16b+', 'nk_cd32+', 'nk_cd64+',
    'nk_fcrl3+', 'nk_fcrl5+', 'nk_fcrl6+',
    'nkp30+|nk', 'nkp44+|nk', 'nkp46+|nk',
    'nkg2a+|nk', 'nkg2c+|nk', 'nkg2d+|nk',
    'kir2dl1+|nk', 'kir2dl2+|nk', 'kir3dl1+|nk',
    'cd57+|nk', 'pd1+|nk',
]

data_markers_nk = data[list_markers_nk]

# a few aesthetics to show simpler tick labels
mapper = {feat: feat[3:].upper() for feat in list_markers_nk if feat.startswith('nk_')}
mapper.update({feat: feat[:-3].upper() for feat in list_markers_nk if feat.endswith('|nk')})
mapper.update(
    {
        'nk_cd56int|cd16hi': 'CD16+',
        'nk_cd16b+': 'CD16b+',
        'nkp30+|nk': 'NKp30+', 'nkp44+|nk': 'NKp44+', 'nkp46+|nk': 'NKp46+',
        'nk_fcrl3+': 'FcRL3+', 'nk_fcrl5+': 'FcRL5+', 'nk_fcrl6+': 'FcRL6+',
        'kir2dl1+|nk': 'KIR2DL1+', 'kir2dl2+|nk': 'KIR2DL2+',
        'pd1+|nk': 'PD-1+',
    },
)

data_markers_nk = data_markers_nk.rename(columns=mapper)

# data for the box plots in CD56+
data_cd56_box = data_cd56.loc[
    :,
    [
        'cd16b+|cd56', 'cd32+|cd56', 'cd64+|cd56',
        'fcrl3+|cd56', 'fcrl5+|cd56', 'fcrl6+|cd56',
        'nkp30+|cd56', 'nkp44+|cd56', 'nkp46+|cd56',
        'nkg2a+|cd56', 'nkg2c+|cd56', 'nkg2d+|cd56',
        'kir2dl1+|cd56', 'kir2dl2+|cd56', 'kir3dl1+|cd56',
        'cd57+|cd56', 'pd1+|cd56'
    ]
]

# mapper for renaming the columns
mapper = {feat: feat[:-5].upper() for feat in data_cd56_box.columns}
mapper.update(
    {
        'cd16b+|cd56': 'CD16b+', 'pd1+|cd56': 'PD-1+',
        'nkp30+|cd56': 'NKp30+', 'nkp44+|cd56': 'NKp44+', 'nkp46+|cd56': 'NKp46+',
        'fcrl3+|cd56': 'FcRL3+', 'fcrl5+|cd56': 'FcRL5+', 'fcrl6+|cd56': 'FcRL6+',
        'kir2dl1+|cd56': 'KIR2DL1+', 'kir2dl2+|cd56': 'KIR2DL2+',
    }
)
data_cd56_box.rename(columns=mapper, inplace=True)

data_cd56_box.insert(0, 'CD16+', data_cd56['cd16hi|cd56'] + data_cd56['cd16int|cd56'])

# PCA
data_adcc = pd.merge(data_adcc, allotype, on='donor', how='left')

# for the PCA analysis and bubble plot in G
# we extract the %max ADCC from metaanalysis
data_top  = data_adcc[['donor', 'top']]
# and finally merge flow cytometry, ADCC and haplotype together -> data
data = reduce(
    lambda df1, df2: pd.merge(df1, df2, on='donor'),
    [
        data_cd56,
        data_top,
        allotype
    ]
)
data.rename(columns={'top': 'max % ADCC'}, inplace=True) # best for the legend title
data.set_index('donor', inplace=True, drop=True)


#%%
### PREPARATION OF THE PCA DATA POINTS ###

# extraction of the feat from panel B (NCR/KIR/KLR)
feat = [
    'nkp30+|cd56', 'nkp44+|cd56', 'nkp46+|cd56',
    'nkg2a+|cd56', 'nkg2c+|cd56', 'nkg2d+|cd56',
    'kir2dl1+|cd56', 'kir2dl2+|cd56', 'kir3dl1+|cd56',
]
X = data_cd56[feat]

pca = PCA(n_components=2)
comp = pca.fit_transform(X)

pca_data = pd.DataFrame(
    data=comp,
    columns=['PC1', 'PC2'],
    index=X.index
)
pca_data.index = data_cd56.index
pca_data = pd.merge(
    pca_data, data,
    left_index=True,
    right_index=True
) # merge on both indices


#%%
### FIGURE ###
fig = plt.figure(figsize = (15,5))

### SUBPLOT NKs

plt.subplot(131)
sns.stripplot(
    data=data_markers_nk,
    size=5,
    alpha=.8,
    palette=sns.color_palette('hls', 18),
) # seaborn consideres data are in the wide format here

sns.boxplot(
    data=data_markers_nk,
    color='white',
    fliersize=0, # there is a stripplot anyway
    showfliers=False,
    boxprops={'facecolor':'none', 'edgecolor':'grey'},
    medianprops={'color':'grey'},
    whiskerprops={'color':'grey'},
    capprops={'color':'grey'},
)

plt.ylabel(
    r'% of NK cells',
    fontdict={'size': 10, 'weight': 'bold'}
)
plt.ylim((0,100))
plt.yticks(size=8)
plt.xticks(rotation=90, size=8)
plt.title("Markers expressed on NK cells in PBMCs", fontweight="bold")


### SUBPLOT CD56+
plt.subplot(132)

sns.stripplot(
    data=data_cd56_box,
    size=5,
    alpha=.8,
    palette=sns.color_palette('hls', 18),
) # seaborn consideres data are in the wide format

sns.boxplot(
    data=data_cd56_box,
    color='white',
    fliersize=0, # there is a stripplot anyway
    showfliers=False,
    boxprops={'facecolor':'none', 'edgecolor':'grey'},
    medianprops={'color':'grey'},
    whiskerprops={'color':'grey'},
    capprops={'color':'grey'},
)

plt.ylabel(
    r'% of CD56$^+$ cells',
    fontdict={'size': 10, 'weight': 'bold'}
)
plt.ylim((0,100))
plt.yticks(size=8)
plt.xticks(rotation=90, size=8)

plt.title("Markers expressed on isolated CD56$^+$ cells", fontweight="bold")


### SUBPLOT G - PCA NCR/KIR/KLR features on total CD56
plt.subplot(133)

sns.scatterplot(
    pca_data.loc[:, 'PC1'],
    pca_data.loc[:, 'PC2'],
    alpha=.7,
    size=pca_data['max % ADCC'],
    sizes=(25, 250),
    hue=pca_data['FCGR3A'],
)

for donor in pca_data.index:
    plt.annotate(
        donor,
        (
            pca_data.loc[donor, 'PC1'] + .5,
            pca_data.loc[donor, 'PC2']
        ),
        fontsize=7,
    )

plt.xlabel(
    f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)",
    fontdict={'size': 10, 'weight': 'bold'}
)
plt.ylabel(
    f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)",
    fontdict={'size': 10, 'weight': 'bold'}
)
plt.xticks(size=8)
plt.yticks(size=8)

plt.title("Principal Component Analysis", fontweight="bold")

plt.legend(
    loc='upper right',
    fontsize=8,
    title=None
)

plt.tight_layout()
sns.despine()

plt.savefig('../figures/figure3.svg')

# %%
