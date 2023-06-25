import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import matplotlib.pylab as pylab
import numpy as np
from collections import Counter
from shan_scripts import plots_1 as p1
from shan_scripts import plots_2 as p2
from shan_scripts import plots as p

# ==== sizes ==== #
# ‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’.
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 7),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
# pylab.rcParams.update(params)

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
bin_num = 30
projlim = 1
project_width = 0.2

    

# ==== hist figure ==== #
def hist_fig(my_path, shan_location, energy_start, energy_end, presentation=False, case=None):

    # ================ good for specific run ================ #
    # df = pd.read_csv(f'{my_path}/data_frame.csv')
    # ho = np.array(pd.read_csv(f'{my_path}/hist_output.csv'))
    # ht = np.array(pd.read_csv(f'{my_path}/hist_target.csv'))
    # ======================================================== #


    # ================ good for average run ================ #
    df = pd.read_csv(f'{my_path}/data_frame.csv')
    ho = pd.read_csv(f"./{shan_location}/output_std.csv")
    ht = pd.read_csv(f"./{shan_location}/target_std.csv")
    # ======================================================== #
    


    # int((13-10)/((13-1) / 48)) = 12
    # ho_last_12 = np.expand_dims(ho[:, -12:].sum(axis=1), axis=1)
    # ht_last_12 = np.expand_dims(ht[:, -12:].sum(axis=1), axis=1)

    # ho_new = np.concatenate((ho[:,:-12], ho_last_12), axis=1)
    # ht_new = np.concatenate((ht[:,:-12], ht_last_12), axis=1)

    # ho = ho_new
    # ht = ht_new

    energies = np.linspace(energy_start, energy_end, ho.sum(axis=0).shape[0])
    width = 0.1 if ho.sum(axis=0).shape[0] > 30 else 0.6

    # ================ good for specific run ================ #
    # if presentation:
    #     p.hist(energies, ho.sum(axis=0), ht.sum(axis=0), f'./shan_scripts/multiple_runs/case_{case}/hist.pdf', case)
    #     p.hist(energies, ho.sum(axis=0), ht.sum(axis=0), f'./shan_scripts/multiple_runs/case_{case}/hist.jpeg', case)
    # ======================================================== #


    # ================ good for average run ================ #
    if presentation:
        p.hist(shan_location, energies, ho['output_mean'], ht['target_mean'], f'./{shan_location}/hist.pdf', case)
        p.hist(shan_location, energies, ho['output_mean'], ht['target_mean'], f'./{shan_location}/hist.jpeg', case)
    # ======================================================== #

    
    # ==== (Nout-Ntrue)/Ntrue ==== #

    
    y = (np.array(ho.sum(axis=0)) - np.array(ht.sum(axis=0))) / np.array(ho.sum(axis=0))


    # ==== Projection of last figure ==== #

    final_list = [0] * bin_num  # The 20 here is the bin number - it may be changed of course.
    y = (df['output'] - df['target'])/df['target']
    bin_list = np.linspace(-ylim, ylim, bin_num)  # Generate the bin limits
    binplace = np.digitize(y, bin_list)  # Divide the list into bins
    bin_partition = Counter(binplace)  # Count the number of showers for each bin.
    for k in bin_partition.keys():
        final_list[int(k) - 1] = bin_partition[k]

    # plt.bar(x=np.linspace(-projlim, projlim, bin_num), height=y, label='output', alpha=0.5, width=project_width)

    # plt.ylim(0, 20)

    y[y>0.5] = 0
    yy1, xx = np.histogram(y, bins=30)
    if presentation:
        p.projection(xx, yy1, f'./{shan_location}/projection.pdf')
        p.projection(xx, yy1, f'./{shan_location}/projection.jpeg')
    # ==== output-target ==== #


    x = np.linspace(0, df.shape[0]-1, df.shape[0], dtype='int')


    # ==== (o-t)/t ==== #


    x = np.linspace(0, df.shape[0]-1, df.shape[0], dtype='int')

    if presentation:
        p.tot(df['target'], (df['output'] - df['target'])/df['target'], f'./{shan_location}/tot.pdf')   
        p.tot(df['target'], (df['output'] - df['target'])/df['target'], f'./{shan_location}/tot.jpeg')