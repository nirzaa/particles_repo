from data_loader import EcalDataIO
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from collections import Counter

i = 3
num_classes = 20
flag = 0

edep_file = os.path.join('./data', 'raw', f'signal.al.elaser.IP0{i}.edeplist.mat')
en_file = os.path.join('./data', 'raw', f'signal.al.elaser.IP0{i}.energy.mat')
en_dep = EcalDataIO.ecalmatio(edep_file)  # Dict with 100000 samples {(Z,X,Y):energy_stamp}
energies = EcalDataIO.energymatio(en_file)
total_out = [0] * num_classes

# for i in range(num_classes):
#     out_sum = sum(t[i] for t in energies)
#     total_out[i] = out_sum

# Eliminate multiple numbers of some kind
min_shower_num = 1
max_shower_num = 20
if min_shower_num > 0:
    del_list = []
    for key in energies:
        if len(energies[key]) < min_shower_num or len(energies[key]) >= max_shower_num:
            del_list.append(key)
    for d in del_list:
        del energies[d]
        del en_dep[d]
output_list = []
for key in energies.keys():
    en_list = torch.Tensor(energies[key])
    num_showers = len(en_list)

    ######### Energy bins Generation ########
    if flag == 0:
        max_energy = 10
        final_list = [0] * num_classes  # The 20 here is the bin number - it may be changed of course.
        
        en_list = np.array(en_list.sort().values)[::-1]

        if len(en_list) >= num_classes:
            final_list[:num_classes] = en_list[:num_classes]
        elif len(en_list) < num_classes:
            final_list[:len(en_list)] = en_list
        output_list.append(final_list)
    #==========================================#

    elif flag == 1:
    ######### Energy bins Generation ########
        max_energy = 10
        final_list = [0] * num_classes  # The 20 here is the bin number - it may be changed of course.
        bin_list = np.linspace(0, max_energy, num_classes)  # Generate the bin limits
        binplace = np.digitize(en_list, bin_list)  # Divide the list into bins
        bin_partition = Counter(binplace)  # Count the number of showers for each bin.
        for k in bin_partition.keys():
            final_list[int(k) - 1] = bin_partition[k]
        n = sum(final_list)
        # final_list = [f / n for f in final_list]    # Bin Normalization by sum
        final_list = torch.Tensor(final_list)  # Wrap it in a tensor - important for training and testing.
        output_list.append(final_list)
    #==========================================#

for i in range(num_classes):
    out_sum = sum(t[i] for t in output_list)
    total_out[i] = out_sum

rng = [i+1 for i in range(num_classes)]
plt.figure(figsize=(12, 6))
plt.bar(rng, total_out, label='output', alpha=0.5)
plt.xlabel('bins number for energies')
plt.ylabel('number of particles')
bars = np.linspace(0, max_energy, num_classes)
bars = [float(f'{i:.2f}') for i in bars]
text = 'Bin Energy range [GeV]: \n'
for i in range(num_classes - 1):
    text += f'{i}: {bars[i]:.1f} - {bars[i + 1]:.1f} \n'
plt.text(15.5, 0.015, text,
             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3}, fontsize='x-small')
plt.savefig('./energies_figure')