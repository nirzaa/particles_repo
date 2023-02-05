import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

# ==== sizes ==== #
# ‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’.
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 7),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

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
    plt.title('180 events')
    plt.savefig(f'./presentation/figures/nont{i}.jpg')

    # ==== output-target ==== #

    sns.set_style("darkgrid")
    plt.figure(num=2)
    plt.clf()
    x = np.linspace(0, df.shape[0]-1, df.shape[0], dtype='int')
    plt.scatter(x, df['target'], label='target')
    plt.scatter(x, df['output'], label='output')
    plt.legend()
    plt.xlabel('Number of event')
    plt.ylabel('Number of particles')
    plt.title('180 events')
    plt.savefig(f'./presentation/figures/to{i}.jpg')

    # ==== (o-t)/t ==== #

    sns.set_style("darkgrid")
    plt.figure(num=2)
    plt.clf()
    x = np.linspace(0, df.shape[0]-1, df.shape[0], dtype='int')
    plt.scatter(x, (df['output'] - df['target'])/df['target'])
    plt.xlabel('Number of event')
    plt.ylabel('(output - target)/target')
    plt.title('180 events')
    plt.savefig(f'./presentation/figures/tot{i}.jpg')
