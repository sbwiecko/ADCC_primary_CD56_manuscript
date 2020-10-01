##### FIGURE 4 - HIERARCHICAL CLUSTERING tSNE CD56

# Parts A (markers heatmaps) and B (phenograph clusters) will be created directly
# in FlowJo and saved as a single SVG file. Parts C, D and E will then be saved as
# 2 separated SVG files and then combined to the FlowJo layout in Inkscape

# Figure C - Cumulative barplot from the tSNE phenograph analysis
# Figure D - Dendrogram
# Figure E - Representation of the clusters, allotype and ADCC data


#%%
import numpy as np
import pandas as pd

from functools import reduce # for merging multiple dataframes

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rcParams['mathtext.default'] = 'bf'

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


#%%
### DATA IMPORT ###

# the tSNE cluster data from FlowJo have been preprocessed and saved in a single CSV file
file_tsne = '../data/tSNE_clusters.csv'
data_tsne = pd.read_csv(file_tsne,
                        comment='#',
                        dtype={'donor': str},
                        sep=',')

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
### DATA PROCESSING ###

# for the bubble plot we extract the max % ADCC
data_top  = data_adcc[['donor', 'top']]
# and finally merge flow cytometry, ADCC and haplotype together -> data
data = reduce(lambda df1, df2: pd.merge(df1, df2),
              [data_tsne, data_top, allotype])
data.rename(columns={'top': 'max % ADCC'}, inplace=True) # best for the legend title
data.set_index('donor', inplace=True, drop=True)

# a last reindexing to be done at the end
data_adcc.set_index('donor', drop=True, inplace=True)
data_tsne.set_index('donor', drop=True, inplace=True)
data_tsne.columns.name = 'tSNE cluster'

# hierarchical clustering
Z = linkage(data_tsne, 'ward', optimal_ordering=False)
data_cluster = data_tsne.copy()
data_cluster['cluster'] = fcluster(Z, 32, criterion='distance')


# finally for the subplot representation of the max % ADCC and haplotype in 
# the different tSNE clusters, we extract the dendrogram clusters ID
bubble_data = data_cluster.loc[:, ['cluster']]
bubble_data = pd.merge(bubble_data, data[['max % ADCC', 'FCGR3A']],
                       left_index=True, right_index=True)
# other than clusters 1--5 are singletons are attributed a unique x-coordinate
bubble_data['cluster'] = np.where(bubble_data['cluster'] > 5, 6,
                                  bubble_data['cluster'])
# and we attribute y-coordinate simply corresponding to the donor id
bubble_data['y'] = -bubble_data['max % ADCC'].index.map(int)


#%%
### FIGURE ###
fig = plt.figure(figsize = (8,3.5))

### DENDROGRAM
ax1=plt.subplot(122)

# we try to make the dendrogram a bit sexier than by default...
hierarchy.set_link_color_palette(['purple',
                                  'cornflowerblue',
                                  'limegreen',
                                  'gold',
                                  'tomato'])

# we need context manager to set the linewidth
with plt.rc_context({'lines.linewidth': 2}):
    dend = dendrogram(Z,
                      labels=data_cluster.index,
                      orientation='right',
                      leaf_font_size=7,
                      leaf_rotation=0,
                      above_threshold_color='lightgrey',
                      color_threshold=32,
                      ax=ax1)

# a few more aesthetics
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

plt.xticks(size=7)
plt.xlabel('height',
           fontdict={'size': 8, 'weight': 'bold'})
plt.ylabel(r'donor',
           fontdict={'size': 9, 'weight': 'bold'})

# annotation and color for each group, coordinates set manually
plt.text(23, 290, 'group V',
         color='tomato',
         fontdict={'size': 8, 'weight': 'bold'})
plt.text(30.6, 191, 'group IV',
         color='gold',
         fontdict={'size': 8, 'weight': 'bold'})
plt.text(21.5, 142, 'group III',
         color='limegreen',
         fontdict={'size': 8, 'weight': 'bold'})
plt.text(17, 66, 'group II',
         color='cornflowerblue',
         fontdict={'size': 8, 'weight': 'bold'})
plt.text(25, 30, 'group I',
         color='purple',
         fontdict={'size': 8, 'weight': 'bold'})

plt.tight_layout()

plt.savefig('../figures/figure4PartC.svg')


#%%
### BARPLOT
fig = plt.figure(figsize = (8, 4))
ax2=plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)

# we order the x-ticks acc. to dendrogram so that we can better see the clusters
data_tsne = data_tsne.reindex(np.flip(dend['ivl']))
data_tsne.plot(kind='bar',
               stacked=True,
               width=1,
               colormap = 'tab20',
               rot=60, # ticks rotation
               figsize=(12,6),
               ax=ax2)

plt.ylabel(r'% CD56$^+$ cells in each cluster',
           fontdict={'size': 11, 'weight': 'bold'})
plt.yticks(fontsize=10)
plt.xticks(rotation=60,
           size=9)
plt.xlabel('donor',
           fontdict={'size': 11, 'weight': 'bold'})

plt.legend(bbox_to_anchor=(1.0, 1.02),
           fontsize=6,
           title=' tSNE\ncluster',
           title_fontsize=7)


### BUBBLE-STRIPPLOT
ax3=plt.subplot2grid((3, 3), (1, 2), rowspan=2)

sns.scatterplot(bubble_data.loc[:, 'cluster'], bubble_data.loc[:, 'y'],
                alpha=.7,
                size=bubble_data['max % ADCC'],
                sizes=(10, 350),
                hue=bubble_data['FCGR3A'],
                edgecolor='k',
                ax=ax3)

for donor in bubble_data.index:
    plt.annotate(donor,
                # little x offset
                (bubble_data.loc[donor, 'cluster']+.175,
                 bubble_data.loc[donor, 'y']),
                fontsize=7)

plt.xlabel(r"group",
           fontdict={'size': 11, 'weight': 'bold'})
plt.xticks([1,2,3,4,5,6],
           ['I', 'II', 'III', 'IV', 'V', 'singletons'],
           size=10)

# a few more aesthetics
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(True)
ax3.get_yaxis().set_visible(False)
ax3.spines['left'].set_visible(False)

plt.legend(bbox_to_anchor=(1.05, 1.),
           loc='upper left',
           fontsize=6,
           title=None,
           title_fontsize=7)

plt.tight_layout()

plt.savefig('../figures/figure4PartsDE.svg')


#####