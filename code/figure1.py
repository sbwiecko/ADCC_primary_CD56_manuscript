##### FIGURE 1 - DIVERSITY_IMMUNITY

# A - Diversity of the main immune population, based on experiment fc022-c
# B - Diversity of the main subsets of NK cells, based on experiment fc022-c
# C - Diversity in the expression of FCGRS, NCR, KIR and KLR in the NK cells, 
#     based on experiments fc022-a and fc022-b


#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl


#%%
### Import of the data ###

data = pd.read_csv('../data/data_all_fc022.csv',
                   comment="#",
                   dtype={'donor': str},)
data.set_index('donor', inplace=True, drop=True)


#%%
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
               'kir2dl1+|nk': 'KIR2DL1/S1/3/5+', 'kir2dl2+|nk': 'KIR2DL2/3/S2+',
               'pd1+|nk': 'PD-1+'})

data_markers_nk = data_markers_nk.rename(columns=mapper)


#%%
### Building of the 3 parts of Figure 1 ###

fig = plt.figure(figsize=(4, 8)) # this is A4 format in inches

### subplot A
ax1 = plt.subplot(311)
# matplotlib barplot has no 'stacked' option so we used pandas plot capabilities
data_immune.plot(kind='bar', 
                 colormap='tab10_r',
                 stacked=True,
                 width=1,
                 ax=ax1) 

plt.xticks(size=6,
           rotation=60)
plt.yticks(size=6)
plt.ylabel(r'% of total PBMCs',
           fontdict={'size': 8, 'weight': 'bold'})
plt.xlabel('donor',
           fontdict={'size': 8, 'weight': 'bold'})

# we precise that NK are actually HLA-DR-negative here
list_immune[0] = r"NK (HLA-DR$^-$)|live"
plt.legend(labels=[name[:-5] for name in list_immune],
           #bbox_to_anchor=(1.01,1),
           fontsize=6,
           loc='upper right')


### subplot B
ax2 = plt.subplot(312)
data_nk_subsets.plot(kind='bar',
                     stacked=True,
                     width=.8,
                     ax=ax2)

plt.xticks(size=6,
           rotation=60)
plt.yticks(size=6)
plt.ylabel(r'% of total PBMCs',
           fontdict={'size': 8, 'weight': 'bold'})
plt.xlabel('donor',
           fontdict={'size': 8, 'weight': 'bold'})

plt.legend(labels=[r'CD56$^{\mathrm{lo}}$CD16$^{\mathrm{-}}$',
                   r'CD56$^{\mathrm{dim}}$CD16$^{\mathrm{hi}}$',
                   r'CD56$^{\mathrm{hi}}$CD16$^{\mathrm{-}}$'],
                   #bbox_to_anchor=(1.01,1),
                   fontsize=6)


### subplot C
ax3 = plt.subplot(313)
sns.stripplot(data=data_markers_nk,
              size=4,
              ax=ax3) # seaborn consideres data are in the wide format here

sns.boxplot(data=data_markers_nk,
            color='white',
            fliersize=0, # there is a stripplot anyway
            ax=ax3)

# loop over each box of the axis and attribute black color
for i,artist in enumerate(ax3.artists):
    artist.set_edgecolor('black')

# loop over each line of the axis and attribute balck color
for i in range(len(ax3.lines)):
    line = ax3.lines[i]
    line.set_color('black')

plt.ylabel(r'% of NK cells',
           fontdict={'size': 8, 'weight': 'bold'})
plt.ylim((0,100))
plt.yticks(size=6)
plt.xticks(rotation=90,
           size=6)


# a few additional aesthetics
fig.text(0.02, 0.978, "A", weight="bold", size=16, horizontalalignment='left')
fig.text(0.02, 0.667, "B", weight="bold", size=16, horizontalalignment='left')
fig.text(0.02, 0.359, "C", weight="bold", size=16, horizontalalignment='left')

plt.tight_layout()

plt.savefig('../figures/figure1.pdf')


######