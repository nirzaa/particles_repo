import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm

def means_plotter():
    means = []
    epoch_list = np.linspace(20, 100, 9, dtype='int')
    my_path_fig = os.path.join('./', 'csv_files', 'rel_error_means')

    for epoch in epoch_list:
        my_path = os.path.join('./', 'csv_files', f'epoch_{epoch}', 'data_frame.csv')
        df = pd.read_csv(my_path)
        my_mean = df.rel_error.mean()
        means.append(my_mean)

    plt.figure(num=2, figsize=(12, 6))
    plt.clf()
    plt.scatter(epoch_list, means, label='means')
    plt.legend()
    plt.plot()
    plt.savefig(os.path.join(my_path_fig))

def weights_anal():
    epochs_list = np.linspace(10, 100, 10, dtype='int')
    for epoch_num in tqdm(epochs_list):
        my_path_weights = os.path.join('./', 'csv_files', f'epoch_{epoch_num}')
        if not os.path.exists(my_path_weights):
            os.makedirs(my_path_weights)
        my_path = os.path.join('./', 'saved', 'models', 'my_model', '0202_223508', f'checkpoint-epoch{epoch_num}.pth')
        checkpoint = torch.load(my_path)
        weights = checkpoint['state_dict']
        layers = list(weights.keys())
        layers_weights = list(weights.values())
        for i, layer in enumerate(layers):
            print(f'{i}. layer: {layer}, weights: {layers_weights[i].shape}')
        for i in [78, 77, 66]:
            ln = weights[layers[i]]
            plt.figure(num=0, figsize=(12, 6))
            if i == 78:
                plt.clf()
                plt.plot(ln.cpu())
            elif i == 77:
                plt.clf()
                im = plt.imshow(ln.cpu(), interpolation='nearest', aspect='auto')
                plt.colorbar(im)
            elif i == 66:
                plt.clf()
                data = ln.cpu().reshape(512, 256)
                im = plt.imshow(data, interpolation='nearest', aspect='auto')
                plt.colorbar(im)
            
            plt.savefig(os.path.join(my_path_weights, f'{layers[i]}.png'))

def load_matrix():

    epoch_list = np.linspace(10, 100, 10, dtype='int')

    my_path = './saved/models/matrices'
    for num in epoch_list:
        R = np.random.RandomState(seed=num)
        plt.figure(num=0, figsize=(12, 6))
        plt.clf()
        matrix = torch.load(os.path.join(my_path, f'weights_tensor_{num}epoch.pt'))
        x = np.transpose(matrix.cpu().numpy())
        # x[x<5] = 0
        labels = ~np.all(x == 0, axis=1)
        labels = np.where(labels == True)[0]
        data = x[~np.all(x == 0, axis=1)]
        data = np.transpose(data)
        im = plt.imshow(data, interpolation='none', aspect='auto')
        plt.colorbar(im)
        if len(labels) > 1:
            places = R.choice(len(labels), min(len(labels), 20), replace=False)
            places.sort()
            x_labels = np.linspace(0, len(labels), len(labels), dtype='int')
            x_labels = x_labels[places]
            y_labels = labels[places]
            plt.xticks(x_labels, y_labels, rotation=90)
        plt.savefig(os.path.join(my_path, f'weights_{num}.png'))
    return None

def svd_tensors():
    
    # epoch_list = np.linspace(10, 100, 10, dtype='int')
    epoch_list = np.arange(10, 250, 20, dtype='int')

    matrices_math = './saved/models/matrices'

    # ==== all eigens ===== #

    total_points = []
    for i, num in enumerate(epoch_list):
        df = pd.read_csv(f'./csv_files/epoch_{num}/data_frame.csv')
        targets = np.floor(df.target.to_numpy()).astype(int)
        matrix = torch.load(os.path.join(matrices_math, f'weights_tensor_{num}epoch.pt'))
        matrix = matrix[targets > 15]
        targets = targets[targets > 15]
        
        x = matrix.cpu().numpy()
        data = x
        u, s, vh = np.linalg.svd(data, full_matrices=False)
        print(f'20 first s values for {num} epoch: {s[:20]}\n')
        X = np.dot(u * s, vh)
        total_points.append(X)
    total_points = np.stack(total_points, axis=0)
    my_path = os.path.join('./', 'saved', 'models', 'matrices', '3d_figures', 'all_eigen')
    if not os.path.exists(my_path):
        os.makedirs(my_path)
    for i, num in enumerate(epoch_list):
        fig = plt.figure(num=0, figsize=(12, 6))
        plt.clf()
        plt.imshow(total_points[i], interpolation='none', aspect='auto')
        plt.savefig(os.path.join(my_path, f'{num}_epoch'))
    
    # ==== 3 eigens ===== #

    total_points = []
    for i, num in enumerate(epoch_list):
        df = pd.read_csv(f'./csv_files/epoch_{num}/data_frame.csv')
        targets = np.floor(df.target.to_numpy()).astype(int)
        matrix = torch.load(os.path.join(matrices_math, f'weights_tensor_{num}epoch.pt'))
        matrix = matrix[targets > 15]
        targets = targets[targets > 15]

        x = matrix.cpu().numpy()
        data = x
        u, s, vh = np.linalg.svd(data, full_matrices=False)
        s[3:] = 0
        X = np.dot(u * s, vh)
        total_points.append(X)
    total_points = np.stack(total_points, axis=0)
    my_path = os.path.join('./', 'saved', 'models', 'matrices', '3d_figures', '3_eigen')
    if not os.path.exists(my_path):
        os.makedirs(my_path)
    for i, num in enumerate(epoch_list):
        fig = plt.figure(num=0, figsize=(12, 6))
        plt.clf()
        plt.imshow(total_points[i], interpolation='none', aspect='auto')
        plt.savefig(os.path.join(my_path, f'{num}_epoch'))


    return None

def eigens_scattering():
    # epoch_list = np.linspace(10, 100, 10, dtype='int')
    epoch_list = np.arange(10, 250, 20, dtype='int')

    matrices_math = './saved/models/matrices'

     # ==== 3 eigens scattering u ===== #

    total_points = []
    for i, num in enumerate(epoch_list):
        df = pd.read_csv(f'./csv_files/epoch_{num}/data_frame.csv')
        targets = np.floor(df.target.to_numpy()).astype(int)
        matrix = torch.load(os.path.join(matrices_math, f'weights_tensor_{num}epoch.pt'), map_location=torch.device('cpu'))
        matrix = matrix[targets > 15]
        targets = targets[targets > 15]
        
        x = matrix.cpu().numpy()
        data = x
        u, s, vh = np.linalg.svd(data, full_matrices=False)
        u_sort = u.copy()
        u_sort.sort(axis=1)
        u_sort = u_sort[:,::-1]
        u_scatter = u_sort[:, :3]
        total_points.append(u_scatter)
    total_points = np.stack(total_points, axis=0)
    my_path = os.path.join('./', 'saved', 'models', 'matrices', '3d_figures', 'scattering')
    if not os.path.exists(my_path):
        os.makedirs(my_path)
    for i, num in enumerate(epoch_list):
        fig = plt.figure(num=0, figsize=(12, 6))

        plt.clf()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(total_points[i,:,0], total_points[i,:,1], total_points[i,:,2])
        for j, txt in enumerate(targets):
            # plt.annotate(txt, (total_points[i,j,0], total_points[i,j,1], total_points[i,j,2]))
            ax.text(total_points[i,j,0], total_points[i,j,1], total_points[i,j,2],  '%s' % (txt), size=10, zorder=1, color='k')
        plt.savefig(os.path.join(my_path, f'{num}_epoch_3d'))

        plt.clf()
        plt.scatter(total_points[i,:,0], total_points[i,:,1])
        for j, txt in enumerate(targets):
            plt.annotate(txt, (total_points[i,j,0], total_points[i,j,1]))
        plt.savefig(os.path.join(my_path, f'{num}_epoch_2d_xy'))

        plt.clf()
        plt.scatter(total_points[i,:,0], total_points[i,:,2])
        for j, txt in enumerate(targets):
            plt.annotate(txt, (total_points[i,j,0], total_points[i,j,2]))
        plt.savefig(os.path.join(my_path, f'{num}_epoch_2d_xz'))

        plt.clf()
        plt.scatter(total_points[i,:,1], total_points[i,:,2])
        for j, txt in enumerate(targets):
            plt.annotate(txt, (total_points[i,j,1], total_points[i,j,2]))
        plt.savefig(os.path.join(my_path, f'{num}_epoch_2d_yz'))

def rel_error_mean_variance():
    epoch_list = np.arange(10, 60, 10, dtype='int')

    my_path = os.path.join('./', 'csv_files', '1_class')
    dirlist = [item for item in os.listdir(my_path) if os.path.isdir(os.path.join(my_path, item)) ]
    dirlist.sort()
    with open(os.path.join(my_path, 'stats.txt'), 'w') as f:
        f.write('Stats for our data\n')
        f.write('='*40)
        f.write('\n\n')
    for dir in dirlist:
        df = pd.read_csv(os.path.join(my_path, dir, 'data_frame.csv'))
        output = df.output.sum()
        target = df.target.sum()
        rel_error_N = abs(output-target) / target

        rel_error = df.rel_error
        mean = rel_error.mean()
        std = rel_error.std()
        epoch = dir.split('_')[-1]
        with open(os.path.join(my_path, 'stats.txt'), 'a+') as f:
            f.write(f'\n{epoch} epoch\n')
            f.write('='*30)
            f.write('\n')
            f.write('relative error mean and std for N per event:\n')
            f.write(f'mean: {mean*100:.2f}%, std: {std*100:.2f}%\n')
            f.write(f'relative error for total N: {rel_error_N*100:.2f}%\n')



if __name__ == '__main__':
    # means_plotter()
    # weights_anal()
    # load_matrix()

    # svd_tensors()
    # eigens_scattering()

    rel_error_mean_variance()
