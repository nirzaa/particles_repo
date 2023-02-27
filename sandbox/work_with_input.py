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

def plot_image(ecalimage, name):
    pyplot.style.use("./shan_scripts/luxe.mplstyle")
    fig,ax = pyplot.subplots()
    im = ax.imshow(ecalimage,interpolation='none',origin='lower',cmap='jet')
    ax.grid(False)
    ax.set_xlabel(r'x [index]')
    ax.set_ylabel(r'z [index]')
    fig.colorbar(im, ax=ax, location='bottom', pad=0.25, \
        label=r"$E_\mathregular{dep}$ [GeV]")
    ax.text(5,32,r"$LUXE$ Input Example", \
        verticalalignment='top')
    pyplot.savefig(name, dpi=200, transparent=True)



if __name__ == '__main__':

    name = 'case_5' # case number
    layers = 20

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
            if j[1] < layers: # taking only the pixels that are in the layers we take into account
                try:
                    sum_dict[i] += en_dep[i][j]
                except:
                    sum_dict[i] = en_dep[i][j]
    # en_dep[x[0]][y[0]] # taking the 0 event, and then the 0 pixel value


    tmp = en_dep[key]

    for z, x, y in tmp:
        d_tens[x, y, z] = tmp[(z, x, y)]
    # d_tens = d_tens.unsqueeze(0)  # Only in conv3d

    x = d_tens.sum(axis=1)
    x = np.transpose(x)
    plot_image(x, f'./sandbox/figures/{name}_image')
    
    events_numbers = pd.read_csv('/mnt/sda1/nirz/particles_repo/csv_files/epoch_25/events_numbers.csv', header=None, dtype='int')
    hist_target = pd.read_csv('/mnt/sda1/nirz/particles_repo/csv_files/epoch_25/hist_target.csv', header=None)
    hist_output = pd.read_csv('/mnt/sda1/nirz/particles_repo/csv_files/epoch_25/hist_output.csv', header=None)
    events_numbers = events_numbers.to_numpy().squeeze(axis=1)
    hist_target = hist_target.to_numpy()
    hist_output = hist_output.to_numpy()
    x = list()
    y = list()
    energy_bins = np.linspace(1, 13, 48)
    for i, event in enumerate(events_numbers):
        
        x.append((hist_target[i] * energy_bins).sum())
        y.append((hist_output[i] * energy_bins).sum() / sum_dict[str(x_keys[event])])

        # x.append((hist_target[0] * energy_bins).sum())
        # y.append((hist_target[0] * energy_bins).sum() / sum_dict[str(x_keys[event])])
        
        # x.append(np.array(energies[str(event)]).sum())
        # y.append(np.array(energies[str(event)]).sum() / sum_dict[str(event)])
    p.ratio(x, y, f'./sandbox/figures/{name}_ratio.png')

