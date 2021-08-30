##### FIGURE 3 - CD56/ MARKERS AND ADCC

# Figure A - Box plots for frequency of NK, CD56+lin+ and TCRgd among total PBMCs
# Figure B - Bar plot for the repartition of NK, NKT and CD56+TCRgd among total PBMCs
# Figure C - Bar plot for the expression of CD16hi, CD16int and CD16- in CD56+ cells
# Figure D - Expression of the FCGR and FcRLs,and NCR/KIR/KLR markers in total CD56+ cells
# Figure E - max %ADCC of CD56+ cells in different FCGR3A haplotypes
# Figure F - logEC50 of CD56+ cells in different FCGR3A haplotypes
# Figure G - PCA 'bubble plot' for expression of NCR/KIR/KLR on total CD56+ cells


#%%
import numpy as np
import pandas as pd

from functools import reduce # for merging multiple dataframes

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from pingouin import anova, pairwise_tukey

import matplotlib as mpl
mpl.style.use('default')
mpl.rcParams['mathtext.default'] = 'regular' # we keep rm until legend in C the we will use another trick

# personalized cmap chosen with my daughter Lisa ;-)
cmap = mpl.colors.LinearSegmentedColormap.from_list("",
                    ["cornflowerblue","lightcoral","gainsboro"])

from sklearn.decomposition import PCA


#%%
### DATA IMPORT ###

# flow cytometry data from experiment fc022; only 'nk', 'nkt' and 'tcrgd' and from panel a
file_nktcr = '../data/data_all_fc022.csv'
data_nktcr = pd.read_csv(file_nktcr,
                         comment="#",
                         usecols=['nk_a', 'nkt_a', 'tcrgd_a']) # don't need donor Id here

# flow cytometry data from experiment fc022 gated on CD56+
file_cd56 = '../data/data_all_fc022_cd56.csv'
data_cd56 = pd.read_csv(file_cd56,
                        comment="#",
                        dtype={'donor': str})
data_cd56.set_index('donor', drop=True, inplace=True)

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
# we don't set yet donor as index because the feature will be required for ANOVA


#%%
### PREPARATION OF THE DATA SETS ###

# data for the box plots in A only need to be reordered
data_nktcr = data_nktcr.loc[:,['nk_a', 'nkt_a', 'tcrgd_a']]

# data for the cumulative barplot in B
# first we select the features of interest
data_cd56_bars = data_cd56.loc[:,['cd56+|live','nk|cd56', 'nkt|cd56', 'tcrgd|cd56']]
# then we do some calculations to get the percent among total PBMCs
data_cd56_bars['nk|total']    = data_cd56_bars['nk|cd56'] * \
                                data_cd56_bars['cd56+|live'] / 100
data_cd56_bars['nkt|total']   = data_cd56_bars['nkt|cd56'] * \
                                data_cd56_bars['cd56+|live'] / 100
data_cd56_bars['tcrgd|total'] = data_cd56_bars['tcrgd|cd56'] * \
                                data_cd56_bars['cd56+|live'] / 100
# we clean the original data that won't be used
data_cd56_bars.drop(['cd56+|live', 'nk|cd56', 'nkt|cd56', 'tcrgd|cd56'],
                    axis=1, inplace=True)

# data for the box plots in D
# we keep only the features of interest
data_cd56_box = data_cd56.loc[:,['cd16b+|cd56', 'cd32+|cd56', 'cd64+|cd56',
                                 'fcrl3+|cd56', 'fcrl5+|cd56', 'fcrl6+|cd56',
                                 'nkp30+|cd56', 'nkp44+|cd56', 'nkp46+|cd56',
                                 'nkg2a+|cd56', 'nkg2c+|cd56', 'nkg2d+|cd56',
                                 'kir2dl1+|cd56', 'kir2dl2+|cd56', 'kir3dl1+|cd56',
                                 'cd57+|cd56', 'pd1+|cd56']]
# mapper for renaming the columns
mapper = {feat: feat[:-5].upper() for feat in data_cd56_box.columns}
mapper.update({'cd16b+|cd56': 'CD16b+', 'pd1+|cd56': 'PD-1+',
               'nkp30+|cd56': 'NKp30+', 'nkp44+|cd56': 'NKp44+', 'nkp46+|cd56': 'NKp46+',
               'fcrl3+|cd56': 'FcRL3+', 'fcrl5+|cd56': 'FcRL5+', 'fcrl6+|cd56': 'FcRL6+',
               'kir2dl1+|cd56': 'KIR2DL1+', 'kir2dl2+|cd56': 'KIR2DL2+'})
data_cd56_box.rename(columns=mapper, inplace=True)

# data for the cumulative barplot in C
# first we select the data of interest
data_cd16_bars = data_cd56.loc[:,['cd16int|cd56', 'cd16hi|cd56']]
# then we do some calculations to get the percent among total CD56+
data_cd16_bars['cd16-|cd56'] = 100 - data_cd16_bars['cd16int|cd56'] \
                                     - data_cd16_bars['cd16hi|cd56']
# and finally we reorder the dataframe
data_cd16_bars = data_cd16_bars.loc[:,['cd16hi|cd56', 'cd16int|cd56', 'cd16-|cd56']]

# for the ADCC analysis in E and F we only need to map the allotype
data_adcc = pd.merge(data_adcc, allotype, on='donor', how='left')

# for the PCA analysis and bubble plot in G
# we extract the %max ADCC from metaanalysis
data_top  = data_adcc[['donor', 'top']]
# and finally merge flow cytometry, ADCC and haplotype together -> data
data = reduce(lambda df1, df2: pd.merge(df1, df2, on='donor'),
              [data_cd56, data_top, allotype])
data.rename(columns={'top': 'max % ADCC'}, inplace=True) # best for the legend title
data.set_index('donor', inplace=True, drop=True)

# a last reindexing to be done at the end
data_adcc.set_index('donor', drop=True, inplace=True)


#%%
### PREPARATION OF THE PCA DATA POINTS ###

# extraction of the feat from panel B (NCR/KIR/KLR)
feat = ['nkp30+|cd56', 'nkp44+|cd56', 'nkp46+|cd56',
        'nkg2a+|cd56', 'nkg2c+|cd56', 'nkg2d+|cd56',
        'kir2dl1+|cd56', 'kir2dl2+|cd56', 'kir3dl1+|cd56',]
X = data_cd56[feat]

pca = PCA(n_components=3)
comp = pca.fit_transform(X)

pca_data = pd.DataFrame(data=comp,
                        columns=['PC1', 'PC2', 'PC3'],
                        index=X.index)
pca_data.index = data_cd56.index
pca_data = pd.merge(pca_data, data, left_index=True, right_index=True) # merge on both indices


#%%
### STATISTICAL TESTS ###

# For subplots D and D we perform a one-way ANOVA to test the null hypothesis that
# two or more groups have the same population mean. Here all the samples are independent
# as coming from different FCGR3A haplotype and tested only in one condition

# !pip install openpyxl
ANOVA_top = anova(data=data_adcc,
                  dv='top',          # dependent variable
                  between='FCGR3A')  # between-subject identifier
ANOVA_top.to_excel('../stats/ANOVA_top_figure3.xlsx')

ANOVA_top_posthoc = pairwise_tukey(data=data_adcc,
                                   dv='top',
                                   between='FCGR3A')
ANOVA_top_posthoc.to_excel('../stats/ANOVA_top_posthoc_figure3.xlsx')

ANOVA_ec50= anova(data=data_adcc,
                  dv='EC50',          # dependent variable
                  between='FCGR3A')   # between-subject identifier
ANOVA_ec50.to_excel('../stats/ANOVA_ec50_figure3.xlsx')

ANOVA_ec50_posthoc= pairwise_tukey(data=data_adcc,
                                   dv='EC50',
                                   between='FCGR3A')
ANOVA_ec50_posthoc.to_excel('../stats/ANOVA_ec50_posthoc_figure3.xlsx')


#%%
### FUNCTION FOR PLOTING THE STARS ###

def stars(ax, x1, x2, y, pval=1, t=0, h=0):
    """
    This function plot a line between the columns x1 and x2 of the stripplot
    and/or boxplot at the y-coordinate y and add the text returned by the `stars`
    function and corresponding to the p-value passed and extracted from the ANOVA
    post-hoc table, at a distance t above the line. As an option we can add a height
    h for plotting little ticks at the extremities of the line.
    """

    def asterisks(pval):
        if   pval <= 1e-4 : return '*'*4
        elif pval <= 1e-3 : return '*'*3
        elif pval <= 1e-2 : return '*'*2
        elif pval <= .05  : return '*'
        else              : return 'ns'
    
    ax.plot([x1, x1, x2, x2], [y-h, y, y, y-h], lw=1, color='k')
    ax.text((x1+x2)*.5, y-t,
             asterisks(pval),
             ha='center',
             va='bottom',
             color='red',
             fontdict={'size': 8})


#%%
# A little wrap function for writing axis label subscript in bold
# see https://stackoverflow.com/questions/22569071/matplotlib-change-math-font-size-and-then-go-back-to-default

def wrap_rcparams(f, params):
    def _f(*args, **kw):
        backup = {key:plt.rcParams[key] for key in params}
        plt.rcParams.update(params)
        f(*args, **kw)
        plt.rcParams.update(backup)
    return _f


#%%
### FIGURE ###
fig = px.scatter_3d(
    pca_data.reset_index(),
    x='PC1', y='PC2', z='PC3',
    size='max % ADCC',
    color='FCGR3A',
    #text='donor',
    hover_data={
        'donor' : True,
        'FCGR3A': True,
        'max % ADCC': ':.1f',
        'PC1': False,
        'PC2': False,
        'PC3': False,
    },
    opacity=.85,
    size_max=30,
    )


# fig.update_layout(legend=dict(
#     yanchor="top",
#     y=0.90,
#     xanchor="left",
#     x=0.10,
#     font=dict(size=8)
# ))
fig.update_layout(
    showlegend=False,
    autosize=False,
    width=530,
    height=470,
    margin=dict(
        l=5,
        r=5,
        b=5,
        t=5,
        pad=1,
    )
)
# fig.update_xaxes(automargin=True)
# fig.update_yaxes(automargin=True)

fig.write_html("pca.html")


#####
# %%
