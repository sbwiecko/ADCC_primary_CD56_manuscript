##### FIGURE 5 - ADCC GLYCOVARIANTS

# Figure A - LV boxplot max %ADCC for each variant
# Figure B - LV boxplot EC50 for each variant
# Figure C - Boxplot ratio max % ADCC to TRA
# Figure D - Boxplot difference log10EC50 to TRA


#%%
import numpy as np
import pandas as pd

from pingouin import rm_anova, pairwise_ttests

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.style.use('default')
mpl.rcParams['mathtext.default'] = 'bf'


#%%
### DATA IMPORT ###

# the raw data from experiments tx012-014 were processed as described in Materials and Methods
# and the percent specific release data points compiled into a single file; see also figureS12
data = pd.read_csv('../data/raw_data_tx012-014.csv',
                   comment='#',
                   index_col=0,
                   dtype={'donor': str})

# the meta data from experiments tx012-014 were obtained from figureS12
meta = pd.read_csv('../data/metaanalysis_tx012-014.csv',
                   comment='#',
                   index_col=0,
                   dtype={'donor': str})


#%%
### DATA PROCESSING ###

# first we clean invalid data obtained from fitting, especially with the deglyc
# variant for which no sigmoid curve could fit to the data
ix = meta['top'] > 1000
meta.loc[ix, ['EC50', 'top']] = np.nan

# for the rmANOVA and posthoc multiple comparisions, we need to provide NaN to the data
# points from which a complete set could not be obtained in order to make the tests work
# (NaN auto dropped, see pingouin docs)
for donor in meta['donor'].unique():
    for antibody in meta['antibody'].unique():
        if meta[(meta['donor'] == donor) & (meta['antibody'] == antibody)].empty:
            meta = meta.append({'donor': donor,
                                'antibody': antibody,
                                'EC50': np.nan,
                                'top': np.nan},
                               ignore_index=True)

# OF NOTE, inputation with the mean of the values obtained in the 'deglyc' group has
# also been tested with no major difference in the conclusion


# we need the pivot table for the 2 last subplots, i.e.
# first pivot the data so that we get the mean of the duplicates
# NOTE: need to use a lambda for np.std with ddof=1
pivot = data.pivot_table(values='percent_spe_release',
                         index=['donor', 'antibody', 'conc'],
                         aggfunc=[np.mean, np.std])
pivot.reset_index(inplace=True)
pivot['conc_log10'] = np.log10(pivot['conc'])
# then set the column index to the top level for more conveniance
pivot.columns = pivot.columns.get_level_values(0)


# we do some tricks to get the value of the max % specific release for each donor
# and use it for the calculation of the ratio for each antibody and for each donor
# so easiest was to pivot to get the desired value and then merge back
pivot_meta_top = meta.pivot(index='donor', columns='antibody', values='top')
pivot_top_TRA = pivot_meta_top['TRA'] # reference
meta = pd.merge(meta, pivot_top_TRA, on='donor', how='left')
meta.rename(columns={'TRA': 'TRA_top'}, inplace=True)

# same for the EC50 and we want to compute difference to TRA this time
pivot_meta_EC50 = meta.pivot(index='donor', columns='antibody', values='EC50')
pivot_ec50_TRA = pivot_meta_EC50['TRA'] # reference
meta = pd.merge(meta, pivot_ec50_TRA, on='donor', how='left')
meta.rename(columns={'TRA': 'TRA_ec50'}, inplace=True)

meta["ratio_top"] = meta['top'] / meta['TRA_top']
meta["diff_ec50"] = meta['EC50'] - meta['TRA_ec50']


#%%
### STATISTICAL TESTS ###

# For all subplots A and B we perform a one-way repeated-measure ANOVA.
# Here all the groups are related because all donors have been tested in all the conditions

# !pip install openpyxl
rmANOVA_top = rm_anova(data=meta,
                       dv='top',           # dependent variable
                       within='antibody',  # within-subject identifier
                       subject='donor',    # subject identifier
                       detailed=True)
rmANOVA_top.to_excel('../stats/rmANOVA_top_figure5.xlsx')

rmANOVA_top_posthoc = pairwise_ttests(data=meta,
                                      dv='top',
                                      within='antibody',
                                      subject='donor',
                                      padjust='holm',
                                      nan_policy='pairwise')
rmANOVA_top_posthoc.to_excel('../stats/rmANOVA_top_posthoc_figure5.xlsx')

rmANOVA_ec50= rm_anova(data=meta,
                       dv='EC50',          # dependent variable
                       within='antibody',  # within-subject identifier
                       subject='donor',    # subject identifier
                       detailed=True)
rmANOVA_ec50.to_excel('../stats/rmANOVA_ec50_figure5.xlsx')

rmANOVA_ec50_posthoc= pairwise_ttests(data=meta,
                                      dv='EC50',
                                      within='antibody',
                                      subject='donor',
                                      padjust='holm',
                                      nan_policy='pairwise')
rmANOVA_ec50_posthoc.to_excel('../stats/rmANOVA_ec50_posthoc_figure5.xlsx')


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
    
    ax.plot([x1, x1, x2, x2], [y-h, y, y, y-h], lw=1, color='w') # no line here
    ax.text((x1+x2)*.5, y-t,
             asterisks(pval),
             ha='center',
             va='bottom',
             color='red',
             fontdict={'size': 8})


#%%
### FIGURE ###
fig = plt.figure(figsize=(7.5, 8)) # modified A4 format in inches

# use same color code as in the supp figure; order is important for the generation of cmap
color_antibody = {'TRA': 'tab:red',
                  'G0': 'tab:brown',
                  'G2': 'tab:olive',
                  'ST3': 'tab:purple',
                  'ST6': 'tab:pink',
                  'deglyc': 'tab:blue'}
# Seaborn does not take a Colormap instance as input for .color_palette but hex values
cpal_colors = [mpl.colors.to_hex(col) for col in list(color_antibody.values())]
sns.set_palette(sns.color_palette(cpal_colors, n_colors=6))

### SUBPLOT A - LV BOXPLOT max %ADCC
ax1 = plt.subplot(321)

boxen = sns.boxenplot(x='antibody', y='top', data=meta,
                      order=['TRA', 'G0', 'G2', 'ST3', 'ST6', 'deglyc'],
                      ax=ax1,
                      showfliers=False, # new seaborn option since v0.10.1
                      )
# a little bit of aesthetics
for line in boxen.lines:
    line.set_linewidth(3) # increase the width of the median line
    line.set_color('white')
    line.set_alpha(1)

sns.stripplot(x='antibody', y='top', data=meta,
              order=['TRA', 'G0', 'G2', 'ST3', 'ST6', 'deglyc'],
              color='black',
              size=5,
              alpha=.7,
              dodge=True,
              ax=ax1)

plt.ylabel(r"maximum % ADCC",
           fontdict={'size': 8, 'weight': 'bold'})
plt.ylim((0,90))
plt.yticks([0,20,40,60,80],
           size=6)
plt.xlabel('trastuzumab variant',
           fontdict={'size': 8, 'weight': 'bold'})
plt.xticks(rotation=0,
           size=8)

# addition of the stars based on the multiple comparison tests
stars(ax1,1,1,83,
 pval=rmANOVA_top_posthoc.query("A == 'G0' & B == 'TRA'")['p-corr'].values[0])
stars(ax1,2,2,80,
 pval=rmANOVA_top_posthoc.query("A == 'G2' & B == 'TRA'")['p-corr'].values[0])
stars(ax1,3,3,78,
 pval=rmANOVA_top_posthoc.query("A == 'ST3' & B == 'TRA'")['p-corr'].values[0])
stars(ax1,4,4,75,
 pval=rmANOVA_top_posthoc.query("A == 'ST6' & B == 'TRA'")['p-corr'].values[0])
stars(ax1,5,5,50,
 pval=rmANOVA_top_posthoc.query("A == 'TRA' & B == 'deglyc'")['p-corr'].values[0])


### SUBPLOT B - LV BOXPLOT EC50
ax2 = plt.subplot(322)

boxen = sns.boxenplot(x='antibody', y='EC50', data=meta,
                      order=['TRA', 'G0', 'G2', 'ST3', 'ST6', 'deglyc'],
                      ax=ax2,
                      showfliers=False, # new seaborn option since v0.10.1
                      )
# a little bit of aesthetics
for line in boxen.lines:
    line.set_linewidth(3) # increase the width of the median line
    line.set_color('white')
    line.set_alpha(1)

sns.stripplot(x='antibody', y='EC50', data=meta,
              order=['TRA', 'G0', 'G2', 'ST3', 'ST6', 'deglyc'],
              color='black',
              size=5,
              alpha=.7,
              dodge=True,
              ax=ax2)

plt.ylabel(r"log$_{10}$EC$_{50}$ (µg/mL)",
           fontdict={'size': 8, 'weight': 'bold'})
plt.yticks(size=6)
plt.ylim((-4.4,-.4))
plt.xlabel('trastuzumab variant',
           fontdict={'size': 8, 'weight': 'bold'})
plt.xticks(rotation=0,
           size=8)

# addition of the stars based on the multiple comparison tests
stars(ax2,1,1,-2.25,
 pval=rmANOVA_ec50_posthoc.query("A == 'G0' & B == 'TRA'")['p-corr'].values[0])
stars(ax2,2,2,-2.35,
 pval=rmANOVA_ec50_posthoc.query("A == 'G2' & B == 'TRA'")['p-corr'].values[0])
stars(ax2,3,3,-1.9,
 pval=rmANOVA_ec50_posthoc.query("A == 'ST3' & B == 'TRA'")['p-corr'].values[0])
stars(ax2,4,4,-2.45,
 pval=rmANOVA_ec50_posthoc.query("A == 'ST6' & B == 'TRA'")['p-corr'].values[0])
stars(ax2,5,5,-1.6,
 pval=rmANOVA_ec50_posthoc.query("A == 'TRA' & B == 'deglyc'")['p-corr'].values[0])


### SUBPLOT C - BOXPLOTS RATIO %MAX ADCC
ax3 = plt.subplot(323)

sns.boxplot(x='antibody', y='ratio_top', data=meta,
            order=['TRA', 'G0', 'G2', 'ST3', 'ST6', 'deglyc'],
            fliersize=0, # there is a stripplot anyway
            ax=ax3)
# loop over each box of the axis and attribute black color
for i,artist in enumerate(ax3.artists):
    artist.set_edgecolor('black')
    artist.set_linewidth(1)
# loop over each line of the axis and attribute black color
for line__ in ax3.lines:
    line__.set_color('black')
    line__.set_linewidth(1)

sns.stripplot(x='antibody', y='ratio_top', data=meta,
              order=['TRA', 'G0', 'G2', 'ST3', 'ST6', 'deglyc'],
              color='black',
              size=5,
              alpha=.7,
              dodge=True,
              ax=ax3)

plt.ylabel(r'ratio maximum % ADCC to TRA',
           fontdict={'size': 8, 'weight': 'bold'})
plt.yticks(fontsize=6)
plt.xlabel('trastuzumab variant',
           fontdict={'size': 8, 'weight': 'bold'})
plt.xticks(rotation=0,
           size=8)


### SUBPLOT D - BOXPLOTS DIFF EC50 ADCC
ax4 = plt.subplot(324)

sns.boxplot(x='antibody', y='diff_ec50', data=meta,
            order=['TRA', 'G0', 'G2', 'ST3', 'ST6', 'deglyc'],
            fliersize=0, # there is a stripplot anyway
            ax=ax4)
# loop over each box of the axis and attribute black color
for i,artist in enumerate(ax4.artists):
    artist.set_edgecolor('black')
    artist.set_linewidth(1)
# loop over each line of the axis and attribute black color
for line_ in ax4.lines:
    line = line_
    line.set_color('black')
    line.set_linewidth(1)

sns.stripplot(x='antibody', y='diff_ec50', data=meta,
              order=['TRA', 'G0', 'G2', 'ST3', 'ST6', 'deglyc'],
              color='black',
              size=5,
              alpha=.7,
              dodge=True,
              ax=ax4)

plt.ylabel(r'difference log$_{10}$EC$_{50}$ to TRA',
           fontdict={'size': 8, 'weight': 'bold'})
plt.yticks(fontsize=6)
plt.xlabel('trastuzumab variant',
           fontdict={'size': 8, 'weight': 'bold'})
plt.xticks(rotation=0,
           size=8)


plt.tight_layout()

fig.text(0.010, 0.972, "A", weight="bold", size=16, horizontalalignment='left')
fig.text(0.510, 0.972, "B", weight="bold", size=16, horizontalalignment='left')
fig.text(0.010, 0.635, "C", weight="bold", size=16, horizontalalignment='left')
fig.text(0.510, 0.635, "D", weight="bold", size=16, horizontalalignment='left')

#plt.savefig('../figures/figure5.svg')
plt.savefig('../figures/figure5.pdf')


#####