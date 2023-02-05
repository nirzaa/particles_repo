import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from collections import Counter

# ==== sizes ==== #
# ‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’.
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 7),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

# https://github.com/nirzaa/particles_repo/tree/20-classes/presentation

# slide 1
# 2d_110classes_allz
# https://github.com/nirzaa/particles_nir_repo_new/tree/paper/csv_files/paper/2d_110classes_allz/run_0/epoch_30

# slide 2
# 2d_110classes_10z
# https://github.com/nirzaa/particles_nir_repo_new/tree/paper/csv_files/paper/2d_110classes_10z/run_0/epoch_30

# slide 3
# 3 micron - first 5 layers
# https://github.com/nirzaa/particles_repo/tree/20-classes/csv_files/3%20micron%20-%20first%205%20layers/run_0/epoch_30

# slide 4
# 3 micron - even layers
# https://github.com/nirzaa/particles_repo/tree/20-classes/csv_files/3%20micron%20-%20even%20layers/run_0/epoch_30

# slide 5
# 3 micron - 1 layer
# https://github.com/nirzaa/particles_nir_repo_new/tree/20energies_2d_noise/csv_files/2d_1z/run_0/epoch_30

ylim = 3
bin_num = 1000
projlim = 1
project_width = 0.2

for i in range(1, 6):
    df = pd.read_csv(f'./presentation/df{i}.csv') # data_frame
    ho = np.array(pd.read_csv(f'./presentation/ho{i}.csv')) # hist_output
    ht = np.array(pd.read_csv(f'./presentation/ht{i}.csv')) # hist_target
    energies = np.linspace(0, 13, ho.sum(axis=0).shape[0])
    width = 0.1 if ho.sum(axis=0).shape[0] > 30 else 0.6

    # ==== hist figure ==== #

    sns.set_style("darkgrid")
    plt.figure(num=0)
    plt.clf()
    plt.bar(x=energies, height=ho.sum(axis=0), label='output', alpha=0.5, width=width)
    plt.bar(x=energies, height=ht.sum(axis=0), label='target', alpha=0.5, width=width)
    plt.legend()
    plt.xlabel('Energies [GeV]')
    plt.ylabel('Number of particles')
    plt.title('180 events')
    plt.savefig(f'./presentation/figures/hist{i}.jpg')

    # ==== (Nout-Ntrue)/Ntrue ==== #

    sns.set_style("darkgrid")
    plt.figure(num=1)
    plt.clf()
    y = (ho.sum(axis=0) - ht.sum(axis=0)) / ho.sum(axis=0)
    plt.scatter(energies, y)
    plt.xlabel('Energies [GeV]')
    plt.ylabel('(Nout-Ntrue)/Ntrue')
    plt.ylim(-ylim, ylim)
    plt.title('180 events')
    plt.savefig(f'./presentation/figures/nont{i}.jpg')

    # ==== Projection of last figure ==== #

    final_list = [0] * bin_num  # The 20 here is the bin number - it may be changed of course.
    y = (df['output'] - df['target'])/df['target']
    bin_list = np.linspace(-ylim, ylim, bin_num)  # Generate the bin limits
    binplace = np.digitize(y, bin_list)  # Divide the list into bins
    bin_partition = Counter(binplace)  # Count the number of showers for each bin.
    for k in bin_partition.keys():
        final_list[int(k) - 1] = bin_partition[k]

    sns.set_style("darkgrid")
    plt.figure(num=1)
    plt.clf()
    plt.hist(y, 20, alpha=0.7)
    # plt.bar(x=np.linspace(-projlim, projlim, bin_num), height=y, label='output', alpha=0.5, width=project_width)
    plt.xlabel('(Nout-Ntrue)/Ntrue')
    plt.ylabel('Occurences')
    plt.ylim(0, max(final_list)+1)
    plt.title('180 events')
    plt.savefig(f'./presentation/figures/projection{i}.jpg')

    # ==== output-target ==== #

    sns.set_style("darkgrid")
    plt.figure(num=2)
    plt.clf()
    x = np.linspace(0, df.shape[0]-1, df.shape[0], dtype='int')
    plt.scatter(x, df['target'], label='target')
    plt.scatter(x, df['output'], label='output')
    plt.legend()
    plt.xlabel('Event number')
    plt.ylabel('Number of particles')
    plt.title('180 events')
    plt.savefig(f'./presentation/figures/to{i}.jpg')

    # ==== (o-t)/t ==== #

    sns.set_style("darkgrid")
    plt.figure(num=2)
    plt.clf()
    x = np.linspace(0, df.shape[0]-1, df.shape[0], dtype='int')
    plt.scatter(df['target'], (df['output'] - df['target'])/df['target'])
    plt.xlabel('Number of multicipies')
    plt.ylabel('(output - target)/target')
    plt.title('180 events')
    plt.savefig(f'./presentation/figures/tot{i}.jpg')
