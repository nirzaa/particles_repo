from numpy import dtype
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def means_plotter():
    means = []
    stds = []
    epoch_list = np.linspace(10, 100, 10, dtype='int')
    my_path_fig = os.path.join('./', 'my_data_done', 'rel_error_means')

    for epoch in epoch_list:
        my_path = os.path.join('./', 'my_data_done', f'epoch_{epoch}', 'data_frame.csv')
        df = pd.read_csv(my_path)
        my_mean = df.rel_error.mean()
        std = df.rel_error.std()
        means.append(my_mean)
        stds.append(std)

    plt.figure(num=2, figsize=(12, 6))
    plt.clf()
    plt.scatter(epoch_list, means, label='means')
    plt.errorbar(epoch_list, means, xerr = 0, yerr = stds)
    plt.legend()
    plt.plot()
    plt.savefig(os.path.join(my_path_fig))

if __name__ == '__main__':
    means_plotter()