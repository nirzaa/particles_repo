import sys
import os
my_path = os.path.join('./')
sys.path.append(my_path)
from data_loader import EcalDataIO
import os
import numpy as np
from collections import Counter
import torch
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
 
# en_dep = EcalDataIO.ecalmatio(os.path.join('D:', os.sep, 'local_github', 'particles_nir_repo', 'data', 'raw', 'signal.al.elaser.IP05.edeplist.mat'))  # Dict with 100000 samples {(Z,X,Y):energy_stamp}
# energies = EcalDataIO.xymatio(os.path.join('D:', os.sep, 'local_github', 'particles_nir_repo', 'data', 'raw', 'signal.al.elaser.IP05.energy.mat'))
# energies = EcalDataIO.energymatio(os.path.join('D:', os.sep, 'local_github', 'particles_nir_repo', 'data', 'raw', 'signal.al.elaser.IP05.energy.mat'))
rng_list = list()
sumy_list = list()
mic_list = [3, 5]
for mic in mic_list:
    en_dep = EcalDataIO.ecalmatio(os.path.join('./data', 'raw', f'signal.al.elaser.IP0{mic}.edeplist.mat'))  # Dict with 100000 samples {(Z,X,Y):energy_stamp}
    energies = EcalDataIO.energymatio(os.path.join('./data', 'raw', f'signal.al.elaser.IP0{mic}.energy.mat'))

    if mic == 3:
        my_keys = energies.keys()
        del_list = list()
        for key in my_keys:
            if len(energies[key]) > 10 and len(energies[key]) < 609:
                del_list.append(key)
        for key in del_list:
            del en_dep[key]
            del energies[key]

    if mic == 5:
        my_keys = energies.keys()
        del_list = list()
        for key in my_keys:
            if len(energies[key]) > 10 and len(energies[key]) < 203:
                del_list.append(key)
        for key in del_list:
            del en_dep[key]
            del energies[key]

    en_list = [energy for energy in energies.values()]
    num_classes = 20
    x_lim = 13

    ######### Energy bins Generation ########
    for i, en in enumerate(en_list):
        bin_num = num_classes
        final_list = [0] * bin_num  # The 20 here is the bin number - it may be changed of course.

        # bin_list = np.linspace(0, x_lim, int(bin_num * 0.5))  # Generate the bin limits
        bin_list = np.linspace(0, x_lim, bin_num)  # Generate the bin limits

        # binplace = np.digitize(en_list, bin_list)  # Divide the list into bins
        binplace = np.digitize(en, bin_list)  # Divide the list into bins
        bin_partition = Counter(binplace)  # Count the number of showers for each bin.
        for k in bin_partition.keys():
            final_list[int(k) - 1] = bin_partition[k]
        n = sum(final_list)
        # final_list = [f / n for f in final_list]    # Bin Normalization by sum
        final_list = torch.Tensor(final_list)  # Wrap it in a tensor - important for training and testing.
        if i == 0:
            final = final_list
            final = final.unsqueeze(axis=0)
        else:
            final_list = final_list.unsqueeze(axis=0)
            final = torch.cat((final, final_list), axis=0)
    #########################################
    sumy = final.sum(axis=0)
    sumy = sumy / sumy.sum()
    rng = [i + 1 for i in range(num_classes)]
    rng_list.append(rng)
    sumy_list.append(sumy)
    plt.clf()
    plt.bar(rng, sumy)
    plt.title(f'{mic} micron - num samples: {final.shape[0]}\nnumber of bins: {num_classes}, max energy: {x_lim}')
    plt.savefig(f'./figures/show_before_run - {mic} micron')

plt.figure()
plt.clf()
alphas = [0.3, 0.5]
for i in range(2):
    plt.bar(rng_list[i], sumy_list[i], label=f'{mic_list[i]} micron', alpha=alphas[i])
plt.title('3 micron target vs. 5 micron target')
plt.legend()
plt.savefig(f'./figures/target 3_vs_5')


exit()
