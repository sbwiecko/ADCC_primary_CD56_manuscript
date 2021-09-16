#%%
from functools import reduce # for merging multiple dataframes

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pingouin import pairwise_ttests

import matplotlib
matplotlib.style.use('default')
#rc('text', usetex=False)
matplotlib.rcParams['mathtext.default'] = 'bf'


#%%
### Import of the data ###

data = pd.read_csv(
    '../data/data_all_fc022.csv',
    comment="#",
    dtype={'donor': str},
)
data.set_index('donor', inplace=True, drop=True)

#  raw data from the ADCC experient tx006
file_tx = '../data/metaanalysis_tx006.csv'
data_tx = pd.read_csv(
    file_tx,
    comment='#',
    index_col=[0],
    dtype={'donor': str},
)

# results of the genotyping for all donors
file_allo = '../data/allotypes_pbmcs.csv'
allotype = pd.read_csv(
    file_allo,
    comment='#',
    usecols=['donor', 'FCGR3A'],
    dtype={'donor': str},
)
# we don't set yet donor as index because the feature will be required for ANOVA

# map the FCGR3A haplotype to each donor in the dataset
data_tx = pd.merge(data_tx, allotype, on='donor', how='left')

# flow cytometry raw data for heatmap and correlation analysis
file_fc = '../data/data_all_fc021.csv'
data_fc = pd.read_csv(
    file_fc,
    comment='#',
    dtype={'donor': str},
)

data_ec50_15 = data_tx[(data_tx['E:T']=='15:1')][['donor', 'EC50']]
data_ec50_15.rename(columns={'EC50': 'EC50_15:1'}, inplace=True)
data_ec50_6 = data_tx[(data_tx['E:T']=='6:1')][['donor', 'EC50']] # requested by reviewer#1
data_ec50_6.rename(columns={'EC50': 'EC50_6:1'}, inplace=True)
data_top_30  = data_tx[(data_tx['E:T']=='30:1')][['donor', 'top']]
data_top_30.rename(columns={'top': 'top_30:1'}, inplace=True)
data_top_15  = data_tx[(data_tx['E:T']=='15:1')][['donor', 'top']] # requested by reviewer#1
data_top_15.rename(columns={'top': 'top_15:1'}, inplace=True)
data_bottom=data_tx[(data_tx['E:T']=='30:1')][['donor', 'bottom']]

data2 = reduce(lambda df1, df2: pd.merge(df1, df2, on='donor'),
              [data_fc, data_ec50_15, data_ec50_6,
               data_top_30, data_top_15, data_bottom])

data2.set_index('donor', inplace=True, drop=True)


# %%
### Preparation of the data for the 3 figures ###

### features A
# We select the variables of the main immune subpopulations
list_immune = ['NK cells|live', 'granulocytes|live', 'monocytes|live',
               'DCs|live', 'B cells|live', 'CD8 T cells|live',
               'CD4 T cells|live','CD4 Treg|live', 'others|live']

data_immune = data[list_immune]


### features B
# We select the variables of the 3 major subsets of NK cells
list_nk_subsets = ['cd56lo_cd16-|live', 'cd56dim_cd16hi|live', 'cd56hi_cd16-|live']

data_nk_subsets = data[list_nk_subsets]


### features C
# we keep the FCGR, FcRL, NCR, KIR and KLR markers for NK cells
list_markers_nk = ['nk_cd16b+', 'nk_cd32+', 'nk_cd64+',
                   'nk_fcrl3+', 'nk_fcrl5+', 'nk_fcrl6+',
                   'nkp30+|nk', 'nkp44+|nk', 'nkp46+|nk',
                   'nkg2a+|nk', 'nkg2c+|nk', 'nkg2d+|nk',
                   'kir2dl1+|nk', 'kir2dl2+|nk', 'kir3dl1+|nk',
                   'cd57+|nk', 'pd1+|nk']

data_markers_nk = data[list_markers_nk]

# a few aesthetics to show simpler tick labels
mapper = {feat: feat[3:].upper() for feat in list_markers_nk if feat.startswith('nk_')}
mapper.update({feat: feat[:-3].upper() for feat in list_markers_nk if feat.endswith('|nk')})
mapper.update({'nk_cd16b+': 'CD16b+',
               'nkp30+|nk': 'NKp30+', 'nkp44+|nk': 'NKp44+', 'nkp46+|nk': 'NKp46+',
               'nk_fcrl3+': 'FcRL3+', 'nk_fcrl5+': 'FcRL5+', 'nk_fcrl6+': 'FcRL6+',
               'kir2dl1+|nk': 'KIR2DL1+', 'kir2dl2+|nk': 'KIR2DL2+',
               'pd1+|nk': 'PD-1+'})

data_markers_nk = data_markers_nk.rename(columns=mapper)


data2.drop(
    [
        'nk_fcrl5+', 'nk_cd16b+', 'nk_cd64+', 'nkp44+nk',
        'pd1+nk', 'pd1+nkt', 'cd57+nkt',
        'nkp30+nkt', 'nkp44+nkt', 'nkp46+nkt',
        'nkg2a+nkt', 'nkg2c+nkt', 'nkg2d+nkt',
        'kir2dl1+nkt', 'kir2dl2+nkt', 'kir3dl1+nkt',
        'cd8_cm', 'cd8_e', 'cd8_em', 'cd8_n',
        'cd4_cm', 'cd4_e', 'cd4_em', 'cd4_n',
        'mono_classic', 'mono_non-classic', 'mono_intermed',
        'mdc', 'pdc'
    ],
    axis=1,
    inplace=True,
)

# mapper for making the feature names more explicit
mapper = {feat: feat.split('_')[1] + '|NK' for feat in data_fc.columns 
          if feat.startswith('nk_')}
mapper.update({feat: feat.split('+')[0] + '+|NKT' for feat in data_fc.columns 
               if feat.endswith('+nkt')})
mapper.update({feat: feat.split('+')[0] + '+|NK' for feat in data_fc.columns 
               if feat.endswith('+nk')})
mapper.update({feat: feat.split('+')[1] + '+|lin' for feat in data_fc.columns 
               if feat.startswith('lin+')})
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

data2.rename(columns=mapper, inplace=True) # here we go!

# we put all the feature names in uppercase for clarity on the figure
data2.columns = data2.columns.str.upper()


#%%
### STATISTICAL TESTS ###

# For subplots A and B we perform a Mixed-design ANOVA to look at the interaction 
# between a within-subject factor (here E:T ratio) for which all the donors 
# have been tested and should be considered related (repeated measures ANOVA)
# and a 'between-subject' factor (here the FCGR3A allotype)

MixedANOVA_top_posthoc = pairwise_ttests(
    data=data_tx,
    padjust='fdr_bh',
    dv='top',
    between='FCGR3A',
    within='E:T',
    subject='donor')

MixedANOVA_ec50_posthoc = pairwise_ttests(
    data=data_tx,
    padjust='fdr_bh',
    dv='EC50',
    between='FCGR3A',
    within='E:T',
    subject='donor'
)


#%%
### FUNCTION FOR PLOTING THE ASTERISKS ###

def asterisks(ax, x1, x2, y, pval=1, t=0, h=0):
    """
    This function plot a line between the columns x1 and x2 of the stripplot
    and/or boxplot at the y-coordinate y and add the text returned by the n_asterisks
    function and corresponding to the p-value passed and extracted from the ANOVA
    post-hoc table, at a distance t above the line. As an option we can add a height
    h for plotting little ticks at the extremities of the line.
    """

    def n_asterisks(pval):
        if   pval <= 1e-4 : return '*'*4
        elif pval <= 1e-3 : return '*'*3
        elif pval <= 1e-2 : return '*'*2
        elif pval <= .05  : return '*'
        else              : return 'ns'
    
    ax.plot([x1, x1, x2, x2], [y-h, y, y, y-h], lw=1, color='k')
    ax.text(
        (x1+x2)*.5,
        y+t,
        n_asterisks(pval),
        ha='center',
        va='bottom',
        color='red',
        fontdict={'size': 10}
    )


#%%
### Building of the 3 parts of Figure 1 ###

fig = plt.figure(figsize = (14, 4))

### subplot A
ax1 = plt.subplot(131)
# matplotlib barplot has no 'stacked' option so we used pandas plot capabilities
data_immune.plot(
    kind='bar', 
    colormap='tab10_r',
    stacked=True,
    width=1,
    ax=ax1,
) 
plt.title("Diversity of immune populations in PBMCS", fontweight="bold")
plt.xticks(size=7, rotation=60)
plt.yticks(size=8)
plt.ylim((0,100))
plt.ylabel(
    r'% of total PBMCs',
    fontdict={'size': 10, 'weight': 'bold'},
)
plt.xlabel(
    'Donor',
    fontdict={'size': 10, 'weight': 'bold'},
)

# we precise that NK are actually HLA-DR-negative here
list_immune[0] = r"NK (HLA-DR$^-$)|live"
plt.legend(
    labels=[name[:-5] for name in list_immune],
    #bbox_to_anchor=(1.01,1),
    fontsize=8,
    loc='upper right',
)


### subplot B
ax2 = plt.subplot(132)
data_nk_subsets.plot(
    kind='bar',
    stacked=True,
    width=.8,
    ax=ax2,
)
plt.title("Diversity of sub-populations of NK cells", fontweight="bold")
plt.xticks(size=7, rotation=60)
plt.yticks(size=8)
plt.ylabel(
    r'% of total PBMCs',
   fontdict={'size': 10, 'weight': 'bold'},
)
plt.xlabel(
    'Donor',
    fontdict={'size': 10, 'weight': 'bold'},
)

plt.legend(
    labels=[r'CD56$^{\mathrm{lo}}$CD16$^{\mathrm{-}}$',
    r'CD56$^{\mathrm{dim}}$CD16$^{\mathrm{hi}}$',
    r'CD56$^{\mathrm{hi}}$CD16$^{\mathrm{-}}$'],
    #bbox_to_anchor=(1.01,1),
    fontsize=8,
)


## subplot C
ax3 = plt.subplot(133)
boxen = sns.boxenplot(
    x='E:T',
    y='top',
    data=data_tx,
    hue='FCGR3A',
    ax=ax3,
    k_depth=4,
    showfliers=False, # new seaborn option since v0.10.1
)

for line in boxen.lines:
    line.set_linewidth(3) # increase the width of the median line
    line.set_color('white')
    line.set_alpha(1)

plt.title("ADCC effector functions of whole PBMCs", fontweight="bold")
sns.stripplot(
    x='E:T',
    y='top',
    data=data_tx,
    hue='FCGR3A',
    color='black',
    size=7,
    alpha=.8,
    dodge=True,
    ax=ax3,
)

plt.xticks(rotation=0, size=9)
plt.yticks([0, 50, 100], size=8)
plt.ylabel(
    r"maximum % ADCC",
    fontdict={'size': 10, 'weight': 'bold'},
)
plt.ylim((0,152))  # need some space for the asterisks
plt.xlabel(
    'E:T ratio',
    fontdict={'size': 10, 'weight': 'bold'},
)

handles, labels = ax3.get_legend_handles_labels()
plt.legend(
    handles[0:3], labels[0:3],
    fontsize=6,
    title='FCGR3A',
    title_fontsize=8,
    loc='lower right',
    bbox_to_anchor=(1, .57),
)

# adding the asterisks resulting for ANOVA post-hoc tests (MixedANOVA_top_posthoc)
asterisks(ax3, 0, 1, 120, t=-.5,
 pval=MixedANOVA_top_posthoc.query("A == '15:1' & B == '30:1'")['p-corr'].values[0])
asterisks(ax3, 0, 2, 131, t=-.5,
 pval=MixedANOVA_top_posthoc.query("A == '30:1' & B == '6:1'")['p-corr'].values[0])
asterisks(ax3, 0, 3, 141, t=-.5,
 pval=MixedANOVA_top_posthoc.query("A == '30:1' & B == '3:1'")['p-corr'].values[0])
asterisks(ax3, 1, 2, 103, t=-.5,
 pval=MixedANOVA_top_posthoc.query("A == '15:1' & B == '6:1'")['p-corr'].values[0])
asterisks(ax3, 1, 3, 114, t=-.5,
 pval=MixedANOVA_top_posthoc.query("A == '15:1' & B == '3:1'")['p-corr'].values[0])
asterisks(ax3, 2, 3, 75,  t=-.5,
 pval=MixedANOVA_top_posthoc.query("A == '3:1' & B == '6:1'")['p-corr'].values[0])


plt.tight_layout()
sns.despine()

plt.savefig('../figures/figure1.svg')


######
# %%
