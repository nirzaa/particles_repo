import os
import h5py
import numpy as np
import torch
# from torchsummary import summary
from torchinfo import summary
from torchvision import models
import model.model as module_arch
import sys
import time
from model import model
import random
from sandbox import loss_analyzer as la
from sandbox import presentation_analyzer as pa
import rel_epochs as re
import pandas as pd

def analyze(model, input_shape, num_runs, folder_name, epoch_nums):

    output_list = list()
    target_list = list()
    rel_error_list = list()
    print_path = os.path.join(folder_name, 'stats.txt')
    with open(print_path, 'w') as f:
        f.write('Stats for our data\n')
        f.write('='*40)
        f.write('\n\n')
        
    with open(print_path, "a+", encoding="utf-8") as log_file:
        sys.stdout = log_file
        print('Model architecture')
        print('='*40)
        summary(model, input_size=input_shape, col_names=('input_size', 'output_size'))
        print('\n\nThe Results:')
        print('='*50)
        print()
    for run_num in range(num_runs):
        with open(os.path.join(folder_name, 'stats.txt'), 'a+') as f:
            f.write(f'run {run_num}\n')
            f.write('='*50)
            f.write('\n')
        my_path = f'{folder_name}/run_{run_num}'
        # for epoch_num in np.linspace(10, epoch_nums, int(epoch_nums / 10), dtype='int'):
        
        # epochs_list = np.append([0], np.linspace(10, epoch_nums, int(epoch_nums / 5)-1, dtype='int'))
        epochs_list = np.linspace(5, epoch_nums, int(epoch_nums / 5), dtype='int')
        
        
        # epochs_list = np.linspace(10, epoch_nums, int(epoch_nums / 5)-1, dtype='int')
        for epoch_num in epochs_list:
            with h5py.File(os.path.join(my_path, f'epoch_{epoch_num}', 'data.h5'), 'r') as hf:
                output = np.array(hf.get('dataset_1'))
                target = np.array(hf.get('dataset_2'))
                rel_error = (output.sum()-target.sum()) / target.sum() * 100
                with open(os.path.join(folder_name, 'stats.txt'), 'a+') as f:
                    f.write(f'The average results for {epoch_num} epoch\n')
                    f.write('='*50)
                    f.write(f'\nThe output average number of particles per event is: {output.mean():.2f}')
                    f.write(f'\nThe target average number of particles per event is: {target.mean():.2f}')
                    f.write(f'\nthe output N value is: {output.sum()}'
                    f'\nthe target N value is: {target.sum()}')
                    f.write(f'\nrelative error for total N: {rel_error:.2f}%\n\n')
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




    # rel_error_N = (mean_output.sum()-mean_target.sum()) / mean_target.sum()
    t = mean_target.sum(axis=1)
    o = mean_output.sum(axis=1)
    with open(os.path.join(folder_name, 'stats.txt'), 'a+') as f:
        f.write(f'The average results for all the above epochs\n')
        f.write('='*50)
        f.write(f'\nThe output average number of particles per event is: {o.mean():.2f}')
        f.write(f'\nThe target average number of particles per event is: {t.mean():.2f}')
        f.write(f'\nthe output N value is: {mean_output.sum()}'
        f'\nthe target N value is: {mean_target.sum()}')
        f.write(f'\nrelative error for total N: {my_rel_error_mean:.2f}%, std: {my_rel_error_std*100:.2f}%\n')

def fluctuation_calculator(num_case, epoch):
    output_list = list()
    target_list = list()
    for run in range(10):
        my_path = f'./csv_files/multiple_runs/case_{num_case}/run_{run}/epoch_{epoch}'
        ho = np.array(pd.read_csv(f'{my_path}/hist_output.csv'))
        ht = np.array(pd.read_csv(f'{my_path}/hist_target.csv'))
        output_list.append(ho.sum(axis=0))
        target_list.append(ht.sum(axis=0))
    output_hist = np.stack(output_list)
    target_hist = np.stack(target_list)

    output_std = np.std(output_hist, axis=0)
    target_std = np.std(target_hist, axis=0)

    output_mean = np.mean(output_hist, axis=0)
    target_mean = np.mean(target_hist, axis=0)

    multiply_mean = np.mean(target_hist * output_hist, axis=0)

    df_output = pd.DataFrame({'output_mean': output_mean, 'output_std': output_std, 'multiply_mean':multiply_mean})
    df_target = pd.DataFrame({'target_mean': target_mean, 'target_std': target_std, 'multiply_mean':multiply_mean})

    df_output.to_csv(f"./shan_scripts/multiple_runs/case_{num_case}/output_std.csv", index=False)
    df_target.to_csv(f"./shan_scripts/multiple_runs/case_{num_case}/target_std.csv", index=False)



    return None



if __name__ == '__main__':

    # 1: WOB 20 layers, 2: WB, 20 layers, 3: WB, 10, 4: WB, 5, 5: WB, 1
    num_case = 2
    epochs_every = 5
    total_epochs = 40
    total_runs = 5
    energy_start = 1
    energy_end = 13
    epochs_num = 40
    num_runs = 5
    input_shape = (128,1,110,21)
    location = f'./csv_files/multiple_runs/case_{num_case}'
    my_path = f'./csv_files/multiple_runs/case_{num_case}/run_7/epoch_25'
    model = model.model_2d_48_1(model_type=None, num_classes=None)

    with open('./run.txt', 'r') as f:
        run = int(f.read())
    SEED = run
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

    # fluctuation_calculator(num_case=5, epoch=25)
    
    # analyze(model, input_shape=input_shape, num_runs=num_runs, folder_name=location, epoch_nums=epochs_num)

    re.rel_fig(num_case, epochs_every, total_epochs, total_runs, presentation=True, case=num_case) # relative error vs. epochs

    # pay attention the hist is based on the run mentioned in my_path
    pa.hist_fig(my_path, energy_start, energy_end, presentation=True, case=num_case) # the figures for the presentation

    
    la.calculate_loss(location, num_runs, epochs_num, presentation=True, case=num_case) # loss function vs. epochs

    
    
    



