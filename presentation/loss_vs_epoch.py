import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
# https://github.com/nirzaa/particles_nir_repo_new/blob/20energies_2d_noise/csv_files/2d_10z/stats.txt
# relative error for total N: 8.02%

# ==== sizes ==== #
# ‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’.
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 7),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

sns.set_style("darkgrid")
plt.figure(num=0)
plt.clf()
epochs = np.linspace(0, 100, 10)
rel_error_total_N = [8.02, 1.52, 0.57, 0.87, 0.69, 0.74, 0.77, 0.72, 0.73, 0.73] # in [%]
plt.scatter(epochs, rel_error_total_N)
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Relative error for total number of particles N [%]')
plt.title('Relative error vs. Epochs')
plt.savefig(f'./presentation/figures/relerror_vs_epochs.jpg')

