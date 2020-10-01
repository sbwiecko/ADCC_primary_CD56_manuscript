##### FIGURE SUPP 8 - HEATMAP CD56

# Figure - Heatmap summarizing the correlation between the different markers in CD56 cells


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import matplotlib
matplotlib.style.use('default')
#rc('text', usetex=False)
matplotlib.rcParams['mathtext.default'] = 'bf'


#%%
### DATA IMPORT ###

# flow cytometry data from experiment fc022 gated on CD56+
file_cd56 = '../data/data_all_fc022_cd56.csv'
data_cd56 = pd.read_csv(file_cd56,
                        comment="#",
                        dtype={'donor': str})

# ADCC data from experiment tx007 using isolated CD56+ cells
file_adcc = '../data/metaanalysis_tx007.csv'
data_adcc = pd.read_csv(file_adcc,
                        comment='#',
                        index_col=[0],
                        dtype={'donor': str})

# we don't need more data in this figure


#%%
### DATA PROCESSING ###

# we first merge both dataframes
data = pd.merge(data_cd56, data_adcc, on='donor')

# exclusion of features not relevant for the analysis
data.drop(['cd56+|live', 'cd16+,cd16b+|live', 'cd16+,cd16b-|live', 'pct_cd56'],
           axis=1, inplace=True)
data.set_index('donor', inplace=True)

# mapper for making the feature names more explicit
mapper = {feat: feat.split('|')[0].upper() for feat in data.columns 
          if feat.endswith('|cd56')}
mapper.update({'nkt|cd56':    'CD56+lin+', 
               'tcrgd|cd56':  'TCRγδ',
               'cd16hi|cd56': 'CD16hi+',
               'cd16int|cd56':'CD16int',
               'cd16b+|cd56': 'CD16b+',
               'fcrl3+|cd56': 'FcRL3+',
               'fcrl5+|cd56': 'FcRL5+',
               'fcrl6+|cd56': 'FcRL6+',
               'pd1+|cd56':   'PD-1+',
               'nkp30+|cd56': 'NKp30+',
               'nkp44+|cd56': 'NKp44+',
               'nkp46+|cd56': 'NKp46+',
               'kir2dl1+|cd56': 'KIR2DL1/S1/3/5+',
               'kir2dl2+|cd56': 'KIR2DL2/3/S2+',
               'EC50':        'Log10EC50',
               'top':         'max%ADCC',
               'bottom':      'min%ADCC'})
data.rename(columns=mapper, inplace=True) # here we go!


#%%
### FIGURE ###
fig = plt.figure(figsize=(8,8)) # modified A4 format in inches

# first creating the mask to keep onty the lower-left part of the heatmap
mask = np.zeros_like(data.corr())
mask[np.triu_indices_from(mask)] = True

hm=sns.heatmap(data.corr(method='kendall'),
               cmap='Spectral_r',
               mask=mask,
               square=True,
               xticklabels=True,
               yticklabels=True,
               cbar=True,
               cbar_kws={"shrink": 0.5, 'pad': -.05})

plt.setp(hm.get_xticklabels(), fontsize=8)
plt.setp(hm.get_yticklabels(), fontsize=8)

# use matplotlib.colorbar.Colorbar object
cbar = hm.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=8)
cbar.set_label(r'Kendall rank correlation coefficient (τ)',
               labelpad=-55)

'''
Kendall’s Tau-b coefficient, which is less sensitive to distortions caused by outliers, 
serves as an alternative to Spearman’s Rho correlation coefficient. If there are ties and 
the samples are small, Kendall’s Tau-b coefficient is preferable to Spearman’s Rho correlation 
coefficient. Kendall’s Tau-b coefficient is usually smaller than Spearman’s Rho correlation coefficient
'''

plt.tight_layout()

#plt.savefig('../figures/figureS8.svg')
plt.savefig('../figures/figureS8.pdf')


#####