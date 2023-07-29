import sys
sys.path.append('./')
from data_loader import EcalDataIO
import os
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pyplot as pyplot
from collections import Counter
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':



    noise_file = os.path.join('./', 'data', 'raw', 'fast.elaser_randomised_bg')
    en_dep = loadmat(noise_file)['0']
    en_dep_noise = torch.zeros((110, 11, 21))
    for i in range(en_dep_noise.shape[0]):
        for j in range(en_dep_noise.shape[1]):
            for k in range(en_dep_noise.shape[2]):
                en_dep_noise[i,j,k] = en_dep[k,i,j]

    fname = f'./utils/multiplicity_histogram/'

    # ==== Energy Histogram 3 micron vs 5 micron ==== #

    fig,ax = pyplot.subplots(num=0)
    plt.figure(num=1)
    plt.title('Histogram of energies')
    plt.xlabel('Energy bin - 250[MeV]')
    plt.ylabel('Density')
    bin_num = 48
    for i in [3,5]:
        energy_hist = np.array([0] * bin_num)
        en_dep_file = f'./data/raw/signal.al.elaser.IP0{i}.edeplist.mat'
        en_file = f'./data/raw/signal.al.elaser.IP0{i}.energy.mat'
        en_dep = EcalDataIO.ecalmatio(en_dep_file)  # Dict with 100000 samples {(Z,X,Y):energy_stamp}
        energies = EcalDataIO.energymatio(en_file)
        keys = list(en_dep.keys())
        for key in keys:
            final_list = [0] * bin_num
            d_tens = np.zeros((110, 11, 21))  # Formatted as [x_idx, y_idx, z_idx]
            
            en_list = torch.Tensor(energies[key])
            bin_list = np.arange(1, 13, 0.25)  # Generate the bin limits
            binplace = np.digitize(en_list, bin_list)  # Divide the list into bins
            bin_partition = Counter(binplace)  # Count the number of showers for each bin.
            for k in bin_partition.keys():
                final_list[int(k) - 1] = bin_partition[k]

            energy_hist += final_list
        energy_hist = energy_hist / energy_hist.sum()
        plt.bar(bin_list, energy_hist, label=str(i))
    plt.legend()
    plt.savefig(fname+'_energy_histogram.png')
    print()

    # ==== Energy vs. Multiplicies ==== #

    #     myE = EcalDataIO.energymatio(en_file)
    #     events = list(en_dep.keys())
    #     energies = list()
    #     multiplicties = list()
    #     for event in events:
    #         energies.append(sum(list(myE[event])))
    #         multiplicties.append(len(myE[event]))

    #     energies = np.array(energies) / 1000
    #     ax.scatter(multiplicties,energies, label=str(i))
    #     ax.set_xlabel(r'Multiplicity')
    #     ax.set_ylabel("Energy[GeV]")
    # pyplot.tight_layout()
    # pyplot.legend()
    # pyplot.savefig(fname+'_energyvsmultiplicity.png')