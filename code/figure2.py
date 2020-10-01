##### FIGURE 2 - ADCC PBMCs

# Figure A - Sumarry of the ADCC %max cytotoxicity values using PBMCs at
#            different E:T ratio and grouped by haplotype
# Figure B - Sumarry of the LOG10 ADCC EC50 values using PBMCs at
#            different E:T ratio and grouped by haplotype
# Figure C - Heatmap of the cytometry and ADCC data using PBMCs (Kendall)
# Figure D - Linear regression EC50 vs. %CD56hiCD16-
# Figure E - Linear regression %max cytotoxicity vs. %CD56 total


#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from pingouin import mixed_anova, pairwise_ttests

import matplotlib
matplotlib.style.use('default')
#rc('text', usetex=False)
matplotlib.rcParams['mathtext.default'] = 'bf'

from functools import reduce # for merging multiple dataframes


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

# map the FCGR3A haplotype to each donor in the dataset
data_tx = pd.merge(data_tx, allotype, on='donor', how='left')

# flow cytometry raw data for heatmap and correlation analysis
file_fc = '../data/data_all_fc021.csv'
data_fc = pd.read_csv(file_fc,
                      comment='#',
                      dtype={'donor': str})


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

data = reduce(lambda df1, df2: pd.merge(df1, df2, on='donor'),
              [data_fc, data_ec50_15, data_ec50_6,
               data_top_30, data_top_15, data_bottom])

data.set_index('donor', inplace=True, drop=True)


#%%
### DATA PREPARATION ###

# features exluded from the analysis (%not significant/relevant)
data.drop(['nk_fcrl5+', 'nk_cd16b+', 'nk_cd64+', 'nkp44+nk',
           'pd1+nk', 'pd1+nkt', 'cd57+nkt',
           'nkp30+nkt', 'nkp44+nkt', 'nkp46+nkt',
           'nkg2a+nkt', 'nkg2c+nkt', 'nkg2d+nkt',
           'kir2dl1+nkt', 'kir2dl2+nkt', 'kir3dl1+nkt',
           'cd8_cm', 'cd8_e', 'cd8_em', 'cd8_n',
           'cd4_cm', 'cd4_e', 'cd4_em', 'cd4_n',
           'mono_classic', 'mono_non-classic', 'mono_intermed',
           'mdc', 'pdc'], 
            axis=1, inplace=True)

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
mapper.update({'nk':            'NK',
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
               })

data.rename(columns=mapper, inplace=True) # here we go!

# we put all the feature names in uppercase for clarity on the figure
data.columns = data.columns.str.upper()


#%%
### STATISTICAL TESTS ###

# For subplots A and B we perform a Mixed-design ANOVA to look at the interaction 
# between a within-subject factor (here E:T ratio) for which all the donors 
# have been tested and should be considered related (repeated measures ANOVA)
# and a 'between-subject' factor (here the FCGR3A allotype)

# !pip install openpyxl
MixedANOVA_top = mixed_anova(data=data_tx,
                             dv='top',          # dependent variable
                             between='FCGR3A',  # between-subject identifier
                             within='E:T',      # within-subject identifier
                             subject='donor')   # subject identifier
MixedANOVA_top.to_excel('../stats/MixedANOVA_top_figure2.xlsx')

MixedANOVA_top_posthoc = pairwise_ttests(data=data_tx,
                                         padjust='fdr_bh',
                                         dv='top',
                                         between='FCGR3A',
                                         within='E:T',
                                         subject='donor')
MixedANOVA_top_posthoc.to_excel('../stats/MixedANOVA_top_posthoc_figure2.xlsx')


MixedANOVA_ec50 = mixed_anova(data=data_tx,
                              dv='EC50',         # dependent variable
                              between='FCGR3A',  # between-subject identifier
                              within='E:T',      # within-subject identifier
                              subject='donor')   # subject identifier
MixedANOVA_ec50.to_excel('../stats/MixedANOVA_ec50_figure2.xlsx')

MixedANOVA_ec50_posthoc = pairwise_ttests(data=data_tx,
                                          padjust='fdr_bh',
                                          dv='EC50',
                                          between='FCGR3A',
                                          within='E:T',
                                          subject='donor')
MixedANOVA_ec50_posthoc.to_excel('../stats/MixedANOVA_ec50_posthoc_figure2.xlsx')


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
### FIGURES ###

fig = plt.figure(figsize=(6.5, 9.5)) # modified A4 format in inches
gs = fig.add_gridspec(nrows=11, ncols=2,)

### subplot A
ax1 = fig.add_subplot(gs[0:3, 0])
boxen = sns.boxenplot(x='E:T', y='top', data=data_tx,
                      hue='FCGR3A',
                      ax=ax1,
                      showfliers=False, # new seaborn option since v0.10.1
                      ) 

# a little bit of customization of the boxen plots
for line in boxen.lines:
    line.set_linewidth(2) # increase the width of the median line
    line.set_color('white')
    line.set_alpha(1)

sns.stripplot(x='E:T', y='top', data=data_tx,
              hue='FCGR3A',
              color='black',
              size=5,
              alpha=.8,
              dodge=True,
              ax=ax1)

plt.ylabel(r"maximum % ADCC",
           fontdict={'size': 8, 'weight': 'bold'})
plt.ylim((0,152))  # need some space for the stars
plt.xlabel('E:T ratio',
           fontdict={'size': 8, 'weight': 'bold'})
plt.xticks(rotation=0,
           size=8)
plt.yticks([0, 50, 100],
           size=6)

handles, labels = ax1.get_legend_handles_labels()
plt.legend(handles[0:3], labels[0:3],
           fontsize=6,
           title='FCGR3A',
           title_fontsize=7,
           loc='lower right',
           bbox_to_anchor=(1, .57))

# adding the stars resulting for ANOVA post-hoc tests (MixedANOVA_top_posthoc)
stars(ax1, 0, 1, 120, t=2,
 pval=MixedANOVA_top_posthoc.query("A == '30:1' & B == '15:1'")['p-corr'].values[0])
stars(ax1, 0, 2, 131, t=2,
 pval=MixedANOVA_top_posthoc.query("A == '30:1' & B == '6:1'")['p-corr'].values[0])
stars(ax1, 0, 3, 141, t=2,
 pval=MixedANOVA_top_posthoc.query("A == '30:1' & B == '3:1'")['p-corr'].values[0])
stars(ax1, 1, 2, 103, t=2,
 pval=MixedANOVA_top_posthoc.query("A == '15:1' & B == '6:1'")['p-corr'].values[0])
stars(ax1, 1, 3, 114, t=2,
 pval=MixedANOVA_top_posthoc.query("A == '15:1' & B == '3:1'")['p-corr'].values[0])
stars(ax1, 2, 3, 75, t=2,
 pval=MixedANOVA_top_posthoc.query("A == '6:1' & B == '3:1'")['p-corr'].values[0])


## subplot B
ax2 = fig.add_subplot(gs[0:3, 1])
boxen = sns.boxenplot(x='E:T', y='EC50', data=data_tx,
                      hue='FCGR3A',
                      ax=ax2,
                      showfliers=False, # new seaborn option since v0.10.1
                      )

for line in boxen.lines:
    line.set_linewidth(2) # increase the width of the median line
    #line.set_linestyle(':')
    line.set_color('white')
    line.set_alpha(1)

sns.stripplot(x='E:T', y='EC50', data=data_tx,
              hue='FCGR3A',
              color='black',
              size=5,
              alpha=.8,
              dodge=True,
              ax=ax2)

plt.ylabel(r"log$_{10}$EC$_{50}$ (µg/mL)",
           fontdict={'size': 8, 'weight': 'bold'})
plt.ylim((-3.6,0.3))
plt.xlabel('E:T ratio',
           fontdict={'size': 8, 'weight': 'bold'})
plt.xticks(rotation=0,
           size=8)
plt.yticks([-1, -2, -3],
           size=6)

handles, labels = ax2.get_legend_handles_labels()
plt.legend(handles[0:3], labels[0:3],
           fontsize=6,
           title='FCGR3A',
           title_fontsize=7,
           loc='lower left',
           bbox_to_anchor=(0, .55))

# adding the stars resulting for ANOVA (see the corresponding table)
stars(ax2, 0, 3, 0, t=.06,
 pval=MixedANOVA_ec50_posthoc.query("A == '30:1' & B == '3:1'")['p-corr'].values[0])
stars(ax2, 1, 3, -.3, t=.06,
 pval=MixedANOVA_ec50_posthoc.query("A == '15:1' & B == '3:1'")['p-corr'].values[0])
stars(ax2, 2, 3, -.6, t=.06,
 pval=MixedANOVA_ec50_posthoc.query("A == '6:1' & B == '3:1'")['p-corr'].values[0])
stars(ax2, 1.75, 2.25, -1, t=.06, h=.05,
 pval=MixedANOVA_ec50_posthoc[(MixedANOVA_ec50_posthoc['Contrast'] == 'E:T * FCGR3A'\
               ) & (MixedANOVA_ec50_posthoc['E:T'] == '6:1')].query("A == 'F/F' & B == 'V/V'"\
                   )['p-corr'].values[0])


### subplot C (heatmap)
ax3 = fig.add_subplot(gs[3:9,:])

# creating the mask to keep onty the lower-left part of the heatmap
mask = np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)] = True

hm=sns.heatmap(data.corr(method='kendall'),
               cmap='Spectral_r',
               mask=mask,
               square=False,
               xticklabels=True,
               yticklabels=True,
               cbar=True,
               cbar_kws={"shrink": 0.5, 'pad': -.15},
               ax=ax3)

plt.setp(hm.get_xticklabels(), fontsize=4)
plt.setp(hm.get_yticklabels(), fontsize=4)

# use matplotlib.colorbar.Colorbar object
cbar = ax3.collections[0].colorbar

# here set the labelsize by 20
cbar.ax.tick_params(labelsize=6)
cbar.set_label(r'Kendall rank correlation coefficient (τ)',
               labelpad=-42,
               size=7)
'''
Kendall's Tau-b coefficient, which is less sensitive to distortions caused
by outliers, serves as an alternative to Spearman's Rho correlation coefficient.
If there are ties and the samples are small, Kendall's Tau-b coefficient is
preferable to Spearman's Rho correlation coefficient. Kendall's Tau-b coefficient
is usually smaller than Spearman's Rho correlation coefficient
'''


### subplot D
# parameters of the linear regression
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(
            data['CD56HI,CD16-|NK'], data['LOG10(EC50;6:1)'])

# plot with ρ indicated
ax4 = fig.add_subplot(gs[-2:, 0])
sns.regplot(x='CD56HI,CD16-|NK', y='LOG10(EC50;6:1)', data=data,
            scatter_kws={"color": "black", 's': 22},
            line_kws={'label': f"ρ={r_value1:.2f}"},
            ax=ax4)

ax4.legend(fontsize=7, frameon=False, loc='lower right')
ax4.set_xlabel(r'%CD56$^{hi}$CD16$^-$ of NK cells',
               fontdict={'size': 7, 'weight': 'bold'})
ax4.set_ylabel(r"log$_{10}$EC$_{50}$",   #+"\n(µg/mL)",
               fontdict={'size': 7, 'weight': 'bold'})
ax4.set_ylim((-3.5, -1))
plt.xticks(size=6)
plt.yticks(size=6)


### subplot E
# parameters of the linear regression
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(
            data['CD56TOTAL'], data['MAX_%_ADCC(30:1)'])

# plot with ρ indicated
ax5 = fig.add_subplot(gs[-2:, 1])
sns.regplot(x='CD56TOTAL', y='MAX_%_ADCC(30:1)', data=data,
            scatter_kws={"color": "black", 's': 22},
            line_kws={'label': f"ρ={r_value2:.2f}"},
            ax=ax5)

ax5.legend(fontsize=7, frameon=False, loc='lower right')
ax5.set_xlabel(r'%CD56$^+$ of PBMCs',
               fontdict={'size': 7, 'weight': 'bold'})
ax5.set_ylabel(r'max % ADCC',
               fontdict={'size': 7, 'weight': 'bold'})
ax5.set_ylim((0,120))
plt.xticks(size=6)
plt.yticks(size=6)


# we open a file to export of the P values and other parameter from the regressions
with open("../stats/pval_figure2.txt", 'w') as f:
    f.write("Linear Regression betwenn CD56HI,CD16-|NK and LOG10(EC50;6:1)\n")
    f.write(f"slope={slope1:.1f}, intercept={intercept1:.1f}, r={r_value1:.2f}, P value={p_value1:.4f} and error={std_err1:.2f}\n")
    f.write('-'*80)
    f.write('\n')
    f.write("Linear Regression betwenn CD56TOTAL and MAX_%_ADCC(30:1)\n")
    f.write(f"slope={slope2:.1f}, intercept={intercept2:.1f}, r={r_value2:.2f}, P value={p_value2:.4f} and error={std_err2:.2f}\n")

plt.tight_layout()

# a few additional aesthetic
fig.text(0.054, 0.975, "A", weight="bold", size=16, horizontalalignment='left')
fig.text(0.525, 0.975, "B", weight="bold", size=16, horizontalalignment='left')
fig.text(0.050, 0.715, "C", weight="bold", size=16, horizontalalignment='left')
fig.text(0.057, 0.147, "D", weight="bold", size=16, horizontalalignment='left')
fig.text(0.522, 0.147, "E", weight="bold", size=16, horizontalalignment='left')


plt.savefig('../figures/figure2.pdf') 

#####