##### FIGURE SUPP 12 - ADCC GLYCOVARIANTS

# Each subplot represents the % specific release data points and the 4P fitted plots 
# for each antibody variant and donor tested with one subplot per donor


#%%
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('default')
mpl.rcParams['mathtext.default'] = 'rm'

from lmfit import Model

#%%
### CONSTANTS AND FOLDERS ###

# we create a 'stats\' folder if does not exist
STATS_PATH = os.path.join('..', "stats")
if not os.path.exists(STATS_PATH):
    os.mkdir(STATS_PATH)

# we create a 'reports\' subfolder in 'stats'\ for all the 4PL regression reports if does not exist
REPORTS_PATH = os.path.join(STATS_PATH, 'reports')
if not os.path.exists(REPORTS_PATH):
    os.mkdir(REPORTS_PATH)

# we create a 'metaanalysis_tx012-014.csv' file in the 'stats\' directory
META_PATH = os.path.join(STATS_PATH, f"metaanalysis_tx012-014.csv")


#%%
### DATA IMPORT ###

# the raw data from experiments tx012-014 were processed as described in Materials and
# Methods and the percent specific release data points compiled into a single file
raw_data = pd.read_csv('../data/raw_data_tx012-014.csv',
                       comment='#',
                       index_col=0,
                       dtype={'donor': str})

# the concentration should be in log10 (see 4PL equation)
raw_data['conc'] = raw_data['conc'].apply(np.log10)


#%%
### DATA PROCESSING ###

# we first pivot the data so that we can get the mean of the duplicates data points
pivot = raw_data.pivot_table(values='percent_spe_release',
                             index=['donor', 'antibody', 'conc'],
                             aggfunc=[np.mean, np.std])

# we also set the same multi-index to new data df
data = raw_data.set_index(['donor', 'antibody', 'conc'])
data.sort_index(level='donor', sort_remaining=False, inplace=True) # sort only by donor

 
#%% PREPARATIONS FOR THE REGRESSION ###
def logistic4(x, top, bottom, ec50, hill_slope):
    """4PL equation with bottom != 0"""
    return (bottom + ((top - bottom) / (1 + 10**((ec50 - x) * hill_slope))))

# parameters set for the model
gmodel = Model(logistic4)
fit_params = gmodel.make_params(top=100, bottom=0, ec50=-1, hill_slope=1) # init model parameters

# we also create a global dataframe that will contain the EC50, % max and % min specific release
# values obtained from each individual fitting and will be used in a different figure
meta = pd.DataFrame(columns=['donor', 'antibody', 'EC50', 'top', 'bottom'])

# finally we define the 2 functions used for extracting the fitting params + saving the regression report
def extract_meta(results, donor, antibody):
    """
    extract metadata from the fitting results and save them into the 'meta' dataframe
    """

    global meta
    meta = meta.append({'donor': donor,
                        'antibody': antibody, 
                        'EC50': results.best_values['ec50'], 
                        'top': results.best_values['top'],
                        'bottom': results.best_values['bottom']}, 
                       ignore_index=True)


def save_report(results, donor, antibody):
    """
    save a fitting report for each donor/antibody tested in a dedicated directory
    """

    with open(os.path.join(REPORTS_PATH, f"4PL_report_{donor}_{antibody}.txt"), 'w', encoding='utf-8') as report:
        report.write(f"*** 4PL regression report - donor:{donor} ; antibody:{antibody} ***\n")
        report.write(results.fit_report(show_correl=False))
        report.write("\n"*2)


#%%
### FIGURE ###
fig = plt.figure(figsize=(8.7, 9)) # modified A4 format in inches

# use same color code as in figure 5; order is important for the generation of cmap
color_antibody = {'TRA':   'tab:red',
                  'G0':    'tab:brown',
                  'G2':    'tab:olive',
                  'ST3':   'tab:purple',
                  'ST6':   'tab:pink',
                  'deglyc':'tab:blue'}

# let's loop over all the donors and plot the raw_data + fit a 4PL curve + generating regression report
donor_list = data.index.get_level_values(0).unique()

i=1 # counter for the subplot id

for donor in donor_list:
    plt.subplot(math.ceil(len(donor_list)/5),5,i) # 5 graphs on each row

    # each donor may have a different set of antibodies tested
    antibody_list = data.loc[donor].index.get_level_values(0).unique()

    for antibody in antibody_list:
        idx = (donor, antibody)

        # model fitting for each antibody
        results = gmodel.fit(pivot.loc[idx][('mean', 'percent_spe_release')],
                             x=pivot.loc[idx].index,
                             params=fit_params)

        # plotting
        ax=results.plot_fit(numpoints=50,
                            fit_kws={'lw': 2,
                                     'c': color_antibody[antibody]},
                            datafmt='none')
        
        ax.plot(data.loc[idx].index, 
                data.loc[idx]['percent_spe_release'],
                label=antibody,
                marker='+',
                mec=color_antibody[antibody],
                ms=6,
                lw=0)
        
        plt.title(donor,
                  fontdict=dict(size=7, weight='bold'))
        plt.xlabel(r'$\log_{10}$[antibody (µg/mL)]',
                   fontdict={'size': 7, 'weight': 'normal'})
        plt.xticks(size=6)
        plt.ylabel('%specific release',
                   fontdict={'size': 7, 'weight': 'normal'})
        plt.yticks(size=6)
        plt.ylim(-5, 80)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles=handles[1::3], labels=labels[2::3],
                   title='antibody',
                   fontsize=5,
                   title_fontsize=6)

        # saving the metadata for this donor/antibody pair
        extract_meta(results, *idx)

        # saving the regression report for this donor/antibody pair
        save_report(results, *idx)

    # increment subplot id counter for the next donor
    i+=1

plt.tight_layout()

plt.savefig('../figures/figureS12.pdf')


# and last we save the dataframe into the csv file that will be used for figure5
f = open(META_PATH, 'w')
# we add a comment line
f.write("# raw data points obtained from the ADCC experiments tx012-014 using CD56+ cells \
isolated from the PBMCs of 22 donors and testing different trastuzumab (TRA) glycovariant \
antibodies. Data are EC50, top and bottom (upper and lower asymptotes resp.) extracted from \
the 4PL regression curves for each donor and antibody tested (E:T=5:1)\n")
f.close()
# and append the data
meta.to_csv(path_or_buf=META_PATH, mode='a')


#####