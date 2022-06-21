import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
# from torchsummary import summary
from torchinfo import summary
from torchvision import models
import model.model as module_arch
import sys
import time
from model import model

def analyze(model, input_shape, num_runs):

    output_list = list()
    target_list = list()
    rel_error_list = list()
    print_path = os.path.join('./csv_files', 'stats.txt')
    with open(print_path, 'w') as f:
        f.write('Stats for our data\n')
        f.write('='*40)
        f.write('\n\n')
        files_list = os.listdir(f'./saved/models/new_model/')
        files_list.sort()
        train_folder = files_list[0]
        
    with open(print_path, "a+") as log_file:
        sys.stdout = log_file
        print('Model architecture')
        print('='*40)
        summary(model, input_size=input_shape, col_names=('input_size', 'output_size'))
        print('\n\nThe Results:')
        print('='*50)
        print()
    for run_num in range(num_runs):
        my_path = f'./csv_files/2d_1z/run_{run_num}'
        for epoch_num in np.linspace(10, 100, 10, dtype='int'):
            with h5py.File(os.path.join(my_path, f'epoch_{epoch_num}', 'data.h5'), 'r') as hf:
                output = np.array(hf.get('dataset_1'))
                target = np.array(hf.get('dataset_2'))
                rel_error = (output.sum()-target.sum()) / target.sum()
                with open(os.path.join('./csv_files', 'stats.txt'), 'a+') as f:
                    f.write(f'The average results for {epoch_num} epoch\n')
                    f.write('='*50)
                    f.write(f'\nThe output average number of particles per event is: {output.mean():.2f}')
                    f.write(f'\nThe target average number of particles per event is: {target.mean():.2f}')
                    f.write(f'\nthe output N value is: {output.sum()}'
                    f'\nthe target N value is: {target.sum()}')
                    f.write(f'\nrelative error for total N: {rel_error*100:.2f}%\n\n')
            output_list.append(output)
            target_list.append(target)
            rel_error_list.append(rel_error)
    my_output = np.stack(output_list, axis=0)
    my_target = np.stack(target_list, axis=0)
    my_rel_error = np.stack(rel_error_list, axis=0)
    mean_output = my_output.mean(axis=0)
    mean_target = my_target.mean(axis=0)
    my_rel_error_mean = my_rel_error.mean()
    my_rel_error_std = my_rel_error.std()
    bars = np.linspace(0, 13, mean_output.shape[1])
    bars = [float(f'{i:.2f}') for i in bars]
    text = 'Bin Energy range [GeV]: \n'
    for i in range(mean_output.shape[1] - 1):
        text += f'{i}: {bars[i]:.1f} - {bars[i + 1]:.1f} \n'
    rng = [i + 1 for i in range(mean_output.shape[1])]
    plt.figure(figsize=(12, 6))
    plt.bar(rng, mean_output.sum(axis=0), label='output', alpha=0.5)
    plt.errorbar(rng, mean_output.sum(axis=0), yerr=(1 / np.sqrt(np.abs(mean_output.sum(axis=0)))), fmt="+", color="b")
    plt.bar(rng, mean_target.sum(axis=0), label='true_val', alpha=0.3)
    plt.xlabel('bins number for energies')
    plt.ylabel('number of particles')
    # plt.text(15.5, 0.015, text,
    plt.text(15.5, 0.015, text,
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3}, fontsize='x-small')

    plt.xticks(rng, rotation=65)
    plt.title(f'{len(output)} samples')
    plt.legend()
    plt.savefig(f'./csv_files/binsgraph.png')



    rel_error_N = (mean_output.sum()-mean_target.sum()) / mean_target.sum()
    t = mean_target.sum(axis=1)
    o = mean_output.sum(axis=1)
    with open(os.path.join('./csv_files', 'stats.txt'), 'a+') as f:
        f.write(f'The average results for all the above epochs\n')
        f.write('='*50)
        f.write(f'\nThe output average number of particles per event is: {o.mean():.2f}')
        f.write(f'\nThe target average number of particles per event is: {t.mean():.2f}')
        f.write(f'\nthe output N value is: {mean_output.sum()}'
        f'\nthe target N value is: {mean_target.sum()}')
        f.write(f'\nrelative error for total N: {my_rel_error_mean*100:.2f}%, std: {my_rel_error_std*100:.2f}%\n')


if __name__ == '__main__':
    model = model.model_2d_10(model_type=None, num_classes=None)
    analyze(model, input_shape=(128,1,110,10), num_runs=3)
