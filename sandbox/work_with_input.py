import os
import numpy as np
import h5py
import scipy.io
import sys, os
import numpy, matplotlib
import matplotlib.pyplot as pyplot
sys.path.append('./')
from data_loader import EcalDataIO
# Generate data set
ecalimage = numpy.random.rand(110,20)
ecalimage = numpy.transpose(ecalimage)
from tqdm import tqdm
import pandas as pd
from shan_scripts import plots as p
import torch
from collections import Counter

def plot_image(ecalimage, name):
    pyplot.style.use("./shan_scripts/luxe.mplstyle")
    fig,ax = pyplot.subplots()
    im = ax.imshow(ecalimage,interpolation='none',origin='lower',cmap='jet')
    ax.grid(False)
    ax.set_xlabel(r'x [width]')
    ax.set_ylabel(r'z [depth]')
    fig.colorbar(im, ax=ax, location='bottom', pad=0.25, \
        label=r"$E_\mathregular{dep}$ [GeV]")
    ax.text(5,32,r"$LUXE$ Input Example", \
        verticalalignment='top')
    pyplot.savefig(name, dpi=200, transparent=True)



if __name__ == '__main__':

    name = 'case_2' # case number
    layers = 10
    location = f'./csv_files/kfold5/{name}/run_0/epoch_25'
    # x = np.random.rand(110,20)
    # x = np.transpose(x)
    # plot_image(x)
    i = 3
    # os.system('ls ./data/raw/')
    en_dep_file = f'./data/raw/signal.al.elaser.IP0{i}.edeplist.mat'
    en_file = f'./data/raw/signal.al.elaser.IP0{i}.energy.mat'
    # mat = scipy.io.loadmat(en_dep_file)
    # x = list(mat.keys())[3:]
    en_dep = EcalDataIO.ecalmatio(en_dep_file)  # Dict with 100000 samples {(Z,X,Y):energy_stamp}
    energies = EcalDataIO.energymatio(en_file)
    
    key = list(en_dep.keys())[0]

    d_tens = np.zeros((110, 11, 21))  # Formatted as [x_idx, y_idx, z_idx]

    x = list(en_dep.keys())
    x_keys = x.copy()
    sum_dict = dict()
    for i in tqdm(x):
        y = list(en_dep[i].keys())
        for j in y:
            if j[0] <= layers: # taking only the pixels that are in the layers we take into account
                try:
                    sum_dict[i] += en_dep[i][j]
                except:
                    sum_dict[i] = en_dep[i][j]
    # en_dep[x[0]][y[0]] # taking the 0 event, and then the 0 pixel value

    en_list = torch.Tensor(energies[key])
    num_showers = len(en_list)
    bin_num = num_classes = 48

    final_list = [0] * bin_num  # The 20 here is the bin number - it may be changed of course.
    bin_list = np.linspace(1, 13, bin_num)  # Generate the bin limits
    binplace = np.digitize(en_list, bin_list)  # Divide the list into bins
    bin_partition = Counter(binplace)  # Count the number of showers for each bin.
    for k in bin_partition.keys():
        final_list[int(k) - 1] = bin_partition[k]
    n = sum(final_list)
    # final_list = [f / n for f in final_list]    # Bin Normalization by sum
    final_list = torch.Tensor(final_list)  # Wrap it in a tensor - important for training and testing.
    p.image_hist(f'./sandbox/figures/{name}_hist', final_list, num_events=num_showers)


    tmp = en_dep[key]

    for z, x, y in tmp:
        d_tens[x, y, z] = tmp[(z, x, y)]
    # d_tens = d_tens.unsqueeze(0)  # Only in conv3d

    x = d_tens.sum(axis=1)
    x = np.transpose(x)
    plot_image(x, f'./sandbox/figures/{name}_image')
    
    events_numbers = pd.read_csv(f'{location}/events_numbers.csv', header=None, dtype='int')
    hist_target = pd.read_csv(f'{location}/hist_target.csv', header=None)
    hist_output = pd.read_csv(f'{location}/hist_output.csv', header=None)
    events_numbers = events_numbers.to_numpy().squeeze(axis=1)
    hist_target = hist_target.to_numpy()
    hist_output = hist_output.to_numpy()

    myE = EcalDataIO.energymatio(en_file)
    events = list(en_dep.keys())
    energies = list()
    multiplicties = list()

    for event in events:
        energies.append(sum(list(myE[event])))
        multiplicties.append(len(myE[event]))

    x = list()
    y = list()
    z = list()
    j =  EcalDataIO.energymatio(en_file)
    z_multicipities = list()
    energy_bins = np.linspace(1, 13, 48)
    for i, event in enumerate(events_numbers):
        
        x.append((hist_target[i] * energy_bins).sum())
        y.append((hist_output[i] * energy_bins).sum() / sum_dict[str(x_keys[event])])
        
        z_multicipities.append(len(myE[str(event)]))
        z.append((hist_target[i] * energy_bins).sum() / sum_dict[str(x_keys[event])])

        # x.append((hist_target[0] * energy_bins).sum())
        # y.append((hist_target[0] * energy_bins).sum() / sum_dict[str(x_keys[event])])
        
        # x.append(np.array(energies[str(event)]).sum())
        # y.append(np.array(energies[str(event)]).sum() / sum_dict[str(event)])
    # p.ratio(x, y, f'./sandbox/figures/{name}_ratio.png')
    
    # p.ratio(x, y, f'./sandbox/figures/{name}', location)
    p.ratio(z_multicipities, y, f'./sandbox/figures/{name}', location)

    x_numpy = np.array(x)
    y_numpy = np.array(y)
    bins = 50
    yy1, xx = np.histogram(y_numpy, bins=bins)

    # yy1, xx = np.histogram(y_numpy, bins=1000)

    # p.projection_sandbox(xx, yy1, f'./sandbox/figures/{name}_ratio_projection.png', y_numpy, bins=30)
    
    p.projection_sand(xx, yy1, f'./sandbox/figures/{name}_ratio_projection.png', bins)
    # p.interval_sand(x_numpy, y_numpy, interval=1000, filename=f'./sandbox/figures/{name}_intervals.png')
    p.interval_sand(x_numpy, y_numpy, interval=100, filename=f'./sandbox/figures/{name}_intervals.png', mypath=location)

    

    fig,ax = pyplot.subplots(num=0)

    energies = np.array(energies) / 1000
    ax.scatter(multiplicties,energies, color='k')
    # ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    # ax.set_xlim(-3,3)
    # ax.set_ylim(0,400)
    ax.set_xlabel(r'Multiplicity')
    ax.set_ylabel("Energy[GeV]")
    fname = f'./sandbox/figures/{name}'
    pyplot.tight_layout()
    pyplot.savefig(fname+'_energyvsmultiplicity.png')


    fig,ax = pyplot.subplots(num=1)

    ax.scatter(z_multicipities,z, color='k')
    # ax.legend(loc=(0.625,0.8))  # defined by left-bottom of legend box; in the ratio of figure size
    # ax.set_xlim(-3,3)
    ax.set_ylim(0,400)
    ax.set_xlabel(r'Multiplicity')
    ax.set_ylabel(r'$E_{gen}[GeV] / E^{tot}_{dep}[MeV]$')
    fname = f'./sandbox/figures/{name}'
    pyplot.tight_layout()
    pyplot.savefig(fname+'_ratio_real.png')

