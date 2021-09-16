#%%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from pingouin import pairwise_tukey

import matplotlib
matplotlib.style.use('default')
#rc('text', usetex=False)
matplotlib.rcParams['mathtext.default'] = 'bf'

from functools import reduce # for merging multiple dataframes


#%%
### DATA IMPORT ###

### FIG2
# raw data from the ADCC experient tx006
data_tx = pd.read_csv(
    '../data/metaanalysis_tx006.csv',
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
# we don't set yet donor as index because the feature will be required for ANOVA

# map the FCGR3A haplotype to each donor in the dataset
data_tx = pd.merge(data_tx, allotype, on='donor', how='left')

# flow cytometry raw data for heatmap and correlation analysis
data_fc = pd.read_csv(
    '../data/data_all_fc021.csv',
    comment='#',
    dtype={'donor': str},
)

# we finally include some parameters from ADCC metaanalysis into data_fc -> data
data_ec50_15 = data_tx[(data_tx['E:T']=='15:1')][['donor', 'EC50']]
data_ec50_15.rename(columns={'EC50': 'EC50_15:1'}, inplace=True)
data_ec50_6 = data_tx[(data_tx['E:T']=='6:1')][['donor', 'EC50']] # requested by reviewer#1
data_ec50_6.rename(columns={'EC50': 'EC50_6:1'}, inplace=True)
data_top_30  = data_tx[(data_tx['E:T']=='30:1')][['donor', 'top']]
data_top_30.rename(columns={'top': 'top_30:1'}, inplace=True)
data_top_15  = data_tx[(data_tx['E:T']=='15:1')][['donor', 'top']] # requested by reviewer#1
data_top_15.rename(columns={'top': 'top_15:1'}, inplace=True)
data_bottom=data_tx[(data_tx['E:T']=='30:1')][['donor', 'bottom']]

data = reduce(
    lambda df1, df2: pd.merge(df1, df2, on='donor'),
    [
        data_fc,
        data_ec50_15,
        data_ec50_6,
        data_top_30,
        data_top_15,
        data_bottom
    ]
)

data.set_index('donor', inplace=True, drop=True)


### FIGS7
# ADCC data using CD56+ cells
data_cd56 = pd.read_csv(
        '../data/metaanalysis_tx007.csv',
        comment='#',
        index_col=[0],
        dtype={'donor': str},
)

# ADCC data using PBMCs
# in this figure only EC50 and E:T are required
data_pbmc = pd.read_csv(
        '../data/metaanalysis_tx006.csv',
        comment='#',
        usecols=['donor', 'E:T', 'EC50'],
        dtype={'donor': str},
)


### FIG3B
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
# we don't set yet donor as index because the feature will be required for ANOVA

# for the ADCC analysis in E and F we only need to map the allotype
data_adcc = pd.merge(data_adcc, allotype, on='donor', how='left')


#%%
### DATA PREPARATION ###

### FIG2
# features exluded from the analysis (%not significant/relevant)
data.drop(
    [
        'nk_fcrl5+', 'nk_cd16b+', 'nk_cd64+', 'nkp44+nk',
        'pd1+nk', 'pd1+nkt', 'cd57+nkt',
        'nkp30+nkt', 'nkp44+nkt', 'nkp46+nkt',
        'nkg2a+nkt', 'nkg2c+nkt', 'nkg2d+nkt',
        'kir2dl1+nkt', 'kir2dl2+nkt', 'kir3dl1+nkt',
        'cd8_cm', 'cd8_e', 'cd8_em', 'cd8_n',
        'cd4_cm', 'cd4_e', 'cd4_em', 'cd4_n',
        'mono_classic', 'mono_non-classic', 'mono_intermed',
        'mdc', 'pdc',
    ],
    axis=1,
    inplace=True,
)

# mapper for making the feature names more explicit
mapper = {
    feat: feat.split('_')[1] + '|NK' for feat in data_fc.columns if feat.startswith('nk_')
}
mapper.update(
    {feat: feat.split('+')[0] + '+|NKT' for feat in data_fc.columns if feat.endswith('+nkt')}
)
mapper.update(
    {feat: feat.split('+')[0] + '+|NK' for feat in data_fc.columns if feat.endswith('+nk')}
)
mapper.update(
    {feat: feat.split('+')[1] + '+|lin' for feat in data_fc.columns if feat.startswith('lin+')}
)
# and some manual upgrade
mapper.update(
    {
        'nk':            'NK',
        'cd56hicd16-':   'CD56hi,CD16-|NK', 
        'cd56intcd16hi': 'CD56int,CD16hi|NK',
        'cd56locd16-':   'CD56lo,CD16-|NK',
        'nkg2c+nk':      'NKG2C+|NK',
        'nkg2c+nkt':     'NKG2C+|CD56+lin+', # inappropriate naming of NKT (reviewer#1)
        'nkt_cd16+':     'CD16+|CD56+,CD3+', # inappropriate naming of NKT (reviewer#1)
        'kir2dl1+nk':    'KIR2DL1/S1/3/5|NK', 
        'kir2dl2+nk':    'KIR2DL2/3/S2|NK',
        't_cell':        'T_cells',
        'b_cell':        'B_cells',
        'gr_nk':         'granulo_NK',
        'nkt(cd56)':     'CD56+,lin+',       # inappropriate naming of NKT (reviewer#1)
        'EC50_6:1':      'Log10(EC50;6:1)',
        'EC50_15:1':     'Log10(EC50;15:1)',
        'top_15:1':      'max_%_ADCC(15:1)',
        'top_30:1':      'max_%_ADCC(30:1)',
        'bottom':        'min_%_ADCC',
    }
)

data.rename(columns=mapper, inplace=True) # here we go!

# we put all the feature names in uppercase for clarity on the figure
data.columns = data.columns.str.upper()


### FIGS7
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

### FIG3B
data_adcc.set_index('donor', drop=True, inplace=True)


# %%
### STATISTICAL TESTS ###

# For subplots D and D we perform a one-way ANOVA to test the null hypothesis that
# two or more groups have the same population mean. Here all the samples are independent
# as coming from different FCGR3A haplotype and tested only in one condition

# !pip install openpyxl
ANOVA_top_posthoc = pairwise_tukey(
    data=data_adcc,
    dv='top',
    between='FCGR3A',
)

ANOVA_ec50_posthoc= pairwise_tukey(
    data=data_adcc,
    dv='EC50',
    between='FCGR3A',
)


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
    ax.text(
        (x1+x2)*.5,
        y-t,
        asterisks(pval),
        ha='center',
        va='bottom',
        color='red',
        fontdict={'size': 11},
    )


#%%
### FIGURES ###
fig = plt.figure(figsize = (14,4.5), constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=5)

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(
    data['CD56HI,CD16-|NK'],
    data['LOG10(EC50;6:1)']
)

# plot with ρ indicated
ax1 = fig.add_subplot(gs[:1, :1])
sns.regplot(
    x='CD56HI,CD16-|NK',
    y='LOG10(EC50;6:1)',
    data=data,
    scatter_kws={"color": "black", 's': 30, 'alpha': .8},
    line_kws={
        'label': f"ρ={r_value1:.2f}",
        'color': 'mediumseagreen',
    },
    ax=ax1,
)

ax1.legend(fontsize=8, frameon=False, loc='lower right')
ax1.set_xlabel(
    r'%CD56$^{hi}$CD16$^-$ of NK cells',
    fontdict={'size': 9, 'weight': 'bold'}
)
ax1.set_ylabel(
    r"log$_{10}$EC$_{50}$",   #+"\n(µg/mL)",
    fontdict={'size': 9, 'weight': 'bold'},
)
ax1.set_ylim((-3.5, -1))
plt.xticks(size=8)
plt.yticks(size=8)

plt.title("Correlation analysis (PBMCs)", fontsize=12, fontweight="bold")



slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
    data['CD56TOTAL'],
    data['MAX_%_ADCC(30:1)']
)

# plot with ρ indicated
ax2 = fig.add_subplot(gs[1:, :1])
sns.regplot(
    x='CD56TOTAL',
    y='MAX_%_ADCC(30:1)',
    data=data,
    scatter_kws={"color": "black", 's': 30, 'alpha': .8},
    line_kws={
        'label': f"ρ={r_value2:.2f}",
        'color': 'crimson',
    },
    ax=ax2,
)

ax2.legend(fontsize=7, frameon=False, loc='lower right')
ax2.set_xlabel(
    r'%CD56$^+$ of PBMCs',
    fontdict={'size': 9, 'weight': 'bold'},
)
ax2.set_ylabel(
    r'max % ADCC',
    fontdict={'size': 9, 'weight': 'bold'},
)
ax2.set_ylim((0,120))
plt.xticks(size=8)
plt.yticks(size=8)


### FIGS7
ax3 = fig.add_subplot(gs[:, 1:3])

# we loop over the 4 different E:T ratio and plot the corresponding histogram
for ratio in data_pbmc['E:T'].unique():
    sns.kdeplot(
        data_pbmc[data_pbmc['E:T'] == ratio]['EC50'],
        shade=False,
        color=colors_hist[ratio],
        linewidth=2,
        legend=False,
    )
    
    sns.rugplot(
        data_pbmc[data_pbmc['E:T'] == ratio]['EC50'],
        color=colors_hist[ratio]
    )
sns.kdeplot(
    data_cd56['EC50'],
    shade=True,
    bw_adjust=2.5,
    color='gray',
    linewidth=4,
    legend=False,
)

sns.rugplot(
    data_cd56['EC50'],
    color='grey',
    height=0.04,
)

plt.ylabel(
    '',
    fontdict={'size': 10, 'weight': 'bold'})
ax3.set_yticks([])
plt.xlabel(
    r"log$_{10}$EC$_{50}$ (µg/mL)",
    fontdict={'size': 10, 'weight': 'bold'})

plt.annotate(
    text='isolated CD56+ cells (E:T=5:1)',
    xy=(-2.8, .8),
    xytext=(-2.6, .87),
    fontsize=10,
    fontweight='bold',
    color='gray',
    arrowprops={'arrowstyle': '-|>'},
)
plt.ylim((0,.9))
plt.xlim((-4.5, 0))
plt.xticks(size=11)
plt.title(
    r'Improved ADCC avidity in isolated CD56$^+$ cells',
    fontdict=dict(size=12, weight='bold')
)


# remake the legend to include the results of the paired t-tests
leg = [
    '30:1  (**)',
    '15:1 (***)',
    '6:1  (***)',
    '3:1 (****)'
]

plt.legend(
    leg,
    loc='center right',
    fontsize=9,
    title='PBMCs (E:T)',
    title_fontsize=10,
)


### FIG3B
ax4 = fig.add_subplot(gs[:, 3:4])
boxen1 = sns.boxenplot(
    x='FCGR3A',
    y='top',
    data=data_adcc,
    color='mediumorchid',
    ax=ax4,
    showfliers=False, # new seaborn option since v0.10.1
    k_depth=4,
)

for line in boxen1.lines:
    line.set_linewidth(4) # increase the width of the median line
    line.set_color('white')
    line.set_alpha(1)

sns.stripplot(
    x='FCGR3A',
    y='top',
    data=data_adcc,
    color='black',
    size=8,
    alpha=.8,
    dodge=True,
    ax=ax4,
)

plt.ylabel(
    r"maximum % ADCC",
    fontdict={'size': 10, 'weight': 'bold'},
)
plt.ylim((0,115)) # gives some space for the stars
plt.yticks([0,20,40,60,80,100], size=8)
plt.xlabel(
    'FCGR3A haplotype',
    fontdict={'size': 10, 'weight': 'bold'},
)
plt.xticks(rotation=0, size=10)

plt.title("Upper asymptote", fontweight="bold")

stars(ax4, 0, 1, 99,  t=-1.5, pval=ANOVA_top_posthoc.query("A == 'F/F' & B == 'F/V'")['p-tukey'].values[0])
stars(ax4, 1, 2, 106, t=-1.5, pval=ANOVA_top_posthoc.query("A == 'F/V' & B == 'V/V'")['p-tukey'].values[0])
stars(ax4, 0, 2, 20,  t= 5.5, pval=ANOVA_top_posthoc.query("A == 'F/F' & B == 'V/V'")['p-tukey'].values[0])


ax5 = fig.add_subplot(gs[:, 4:])

boxen2 = sns.boxenplot(
    x='FCGR3A',
    y='EC50',
    data=data_adcc,
    color='crimson',
    ax=ax5,
    showfliers=False, # new seaborn option since v0.10.1
    k_depth=4,
)

for line in boxen2.lines:
    line.set_linewidth(4) # increase the width of the median line
    line.set_color('white')
    line.set_alpha(1)

sns.stripplot(
    x='FCGR3A',
    y='EC50',
    data=data_adcc,
    color='black',
    size=8,
    alpha=.8,
    dodge=True,
    ax=ax5
)

plt.ylabel(
    r"log$_{10}$EC$_{50}$ (µg/mL)",
    fontdict={'size': 10, 'weight': 'bold'},
)
plt.ylim((-3.6, -1.9))
plt.yticks([-3.5, -3.0, -2.5, -2.0], size=8)
plt.xlabel(
    'FCGR3A haplotype',
    fontdict={'size': 10, 'weight': 'bold'},
)
plt.xticks(rotation=0, size=10)

# adding the stars
stars(ax5, 0, 1, -2.20, t=-.02, pval=ANOVA_ec50_posthoc.query("A == 'F/F' & B == 'F/V'")['p-tukey'].values[0])
stars(ax5, 1, 2, -2.35, t=-.02, pval=ANOVA_ec50_posthoc.query("A == 'F/V' & B == 'V/V'")['p-tukey'].values[0])
stars(ax5, 0, 2, -2.05, t=-.02, pval=ANOVA_ec50_posthoc.query("A == 'F/F' & B == 'V/V'")['p-tukey'].values[0])


plt.title("Mid-range concentration", fontweight="bold")


sns.despine()
plt.savefig('../figures/figure2.svg') 

#####
 # %%
