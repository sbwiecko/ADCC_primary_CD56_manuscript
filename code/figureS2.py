##### FIGURE SUPP 2 - DIVERSITY NK CELLS

# Each subplot represents the frequency of the NK cells positive for the 
# marker indicated in title for each donor (complement to Figure 1C)


#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl

import math

#%%
### Import of the data and cleaning ###

data = pd.read_csv("../data/data_all_fc022.csv",
                   comment="#",
                   dtype={'donor': str})

data.set_index('donor', inplace=True)


#%%
### Preparation of the data for the figure ###

# Feature selection (only NK cells)
list_markers = ['nk_cd16b+', 'nk_cd32+', 'nk_cd64+',
                'nk_fcrl3+', 'nk_fcrl5+', 'nk_fcrl6+',
                'nkp30+|nk', 'nkp44+|nk', 'nkp46+|nk',
                'nkg2a+|nk', 'nkg2c+|nk', 'nkg2d+|nk',
                'kir2dl1+|nk', 'kir2dl2+|nk', 'kir3dl1+|nk',
                'cd57+|nk', 'pd1+|nk']

data = data[list_markers]

# mapper for renaming the columns, i.e. removing the prefix 'nk_'...
mapper = {feat: feat[3:].upper() for feat in list_markers if feat.startswith('nk_')}
# ...removing the suffix '|nk' and putting in uppercase...
mapper.update({feat: feat[:-3].upper() for feat in list_markers if feat.endswith('|nk')})
# ...and finally modifying manually a few marker names
mapper.update({'nk_cd16b+': 'CD16b+', 'nkp30+|nk': 'NKp30+', 
               'nkp44+|nk': 'NKp44+', 'nkp46+|nk': 'NKp46+',
               'nk_fcrl3+': 'FcRL3+', 'nk_fcrl5+': 'FcRL5+', 'nk_fcrl6+': 'FcRL6+',
               'kir2dl1+|nk': 'KIR2DL1/S1/3/5+', 'kir2dl2+|nk': 'KIR2DL2/3/S2+',
               'pd1+|nk': 'PD-1+'})

data = data.rename(columns=mapper) # here we go!


#%%
### Building of the figure ###

fig = plt.figure(figsize=(8, 11)) # modified A4 format in inches

# we want to reuse the color set from figure 1C
cmap = mpl.colors.ListedColormap(sns.color_palette("husl", 17))

# now we loop over the list of markers and plot the frequency for each donor
i=1
n=len(data.columns) # total number of markers to plot

for marker in data.columns:
    ax = plt.subplot(math.ceil(n/3),3,i) # 3 columns looks good
    
    data[marker].plot(kind='bar',
                      color=cmap((i-1)/n),
                      width=.6,
                      legend=None,
                      ax=ax)

    plt.title(marker,
              fontdict=dict(size=8, weight='bold'))
    plt.xticks(rotation=90,
               fontsize=5)
    plt.xlabel('')
    plt.yticks(fontsize=6)
    plt.ylabel(r'% of NK cells',
               fontdict={'size': 7, 'weight': 'bold'})
    plt.ylim((0,100))

    i+=1


plt.tight_layout()

#plt.savefig('../figures/figureS2.svg')
plt.savefig('../figures/figureS2.pdf')


#####