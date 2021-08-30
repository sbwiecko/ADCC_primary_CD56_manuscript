##### FIGURE SUPP 3 - GENOTYPING

# A - Dotplot ddPCR donor 043 (F/F) from experiment gn006, well E06
# B - Dotplot ddPCR donor 041 (F/V) from experiment gn006, well C06
# C - Dotplot ddPCR donor 038 (V/V) from experiment gn006, well H01
# D - Summary of the genotyping data for all donors (pie chart) 

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'it'
mpl.rc('text', usetex=False)


#%%
### Import of the data ###

# importing the 3 .csv amplitude files from the samples selected
file_038 = '../data/plate_gn006_190918_H01_Amplitude.csv'
data_038 = pd.read_csv(file_038)
file_041 = '../data/plate_gn006_190918_C06_Amplitude.csv'
data_041 = pd.read_csv(file_041)
file_043 = '../data/plate_gn006_190918_E06_Amplitude.csv'
data_043 = pd.read_csv(file_043)

# this one below is for part D (pie chart)
data_allotype = pd.read_csv('../data/allotypes_pbmcs.csv',
                            comment="#",
                            dtype={'donor': str})


#%%
### Preparation of the data for the 3 subfigures ###
# we sample randomly 30% of the datapoints to make the figures less heavy
data_038 = data_038.sample(frac=.3)
data_041 = data_041.sample(frac=.3)
data_043 = data_043.sample(frac=.3)

# and then we simply take the x,y coordinates from each ddPCR raw data set
x_038, y_038 = data_038['Ch2 Amplitude'], data_038['Ch1 Amplitude']
x_041, y_041 = data_041['Ch2 Amplitude'], data_041['Ch1 Amplitude']
x_043, y_043 = data_043['Ch2 Amplitude'], data_043['Ch1 Amplitude']


#%%
### Building of the 4 parts A,B,C,D ###

fig = plt.figure(figsize=(7, 6)) # this is ca. half a A4 format in inches

### subplot A
ax1 = plt.subplot(221)
ax1.scatter(x_043, y_043,
            c='black',
            s=5,
            alpha=0.15,
            edgecolor='')

ax1.set_title(r"Fcgr3a-559(A;A)",
              size=10,
              style='italic')
ax1.set(xlim=(0,5000), ylim=(0, 12000))
ax1.set_xlabel('VIC amplitude',
               fontdict={'size': 8, 'weight': 'bold'})
ax1.set_ylabel('FAM amplitude',
               fontdict={'size': 8, 'weight': 'bold'})
ax1.tick_params(labelsize=6)

### subplot B
ax2 = plt.subplot(222)
ax2.scatter(x_041, y_041,
            c='black',
            s=5,
            alpha=0.15,
            edgecolor='')

ax2.set_title(r"Fcgr3a-559(A;C)",
              size=10,
              style='italic')
ax2.set(xlim=(0,5000), ylim=(0, 12000))
ax2.set_xlabel('VIC amplitude',
               fontdict={'size': 8, 'weight': 'bold'})
ax2.set_ylabel('FAM amplitude',
               fontdict={'size': 8, 'weight': 'bold'})
ax2.tick_params(labelsize=6)

### subplot C
ax3 = plt.subplot(223)
ax3.scatter(x_038, y_038,
            c='black',
            s=5,
            alpha=0.15,
            edgecolor='')

ax3.set_title(r"Fcgr3a-559(C;C)",
              size=10,
              style='italic')
ax3.set(xlim=(0,5000), ylim=(0, 12000))
ax3.set_xlabel('VIC amplitude',
               fontdict={'size': 8, 'weight': 'bold'})
ax3.set_ylabel('FAM amplitude',
               fontdict={'size': 8, 'weight': 'bold'})
ax3.tick_params(labelsize=6)


### subplot D
# pie chart has no pct methods :-(
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return f"{absolute:d} donors"

ax4 = plt.subplot(224)
data_pie = data_allotype.groupby('allotype')['allotype'].count()
wedges, texts =  ax4.pie(data_pie,
                         wedgeprops=dict(width=.5, edgecolor='w'),
                         labels=data_pie,
                         startangle=0,
                         radius=1)

ax4.legend(wedges, data_pie.index,
           loc='upper left',
           bbox_to_anchor=(-0.2, 1.0),
           fontsize=8,
           title=' Fcgr3a\nallotype',
           title_fontsize=9)

plt.setp(texts, size=10, weight="bold")

# a few additional aesthetic
# fig.text(0.010, 0.95, "A", weight="bold", size=16, horizontalalignment='left')
# fig.text(0.497, 0.95, "B", weight="bold", size=16, horizontalalignment='left')
# fig.text(0.010, 0.47, "C", weight="bold", size=16, horizontalalignment='left')
# fig.text(0.500, 0.47, "D", weight="bold", size=16, horizontalalignment='left')

plt.tight_layout()

plt.savefig('../figures/figureS3.svg')
#plt.savefig('../figures/figureS3.pdf')


#####