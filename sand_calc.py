from black import out
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
import torch

def rely():
    # path = './csv_files/class_2d_epochs_20energies/run_0/epoch_50'
    path = './csv_files/epoch_30'
    df = pd.read_csv(os.path.join(path, 'data_frame.csv'))
    rel_error = df.rel_error

    print(f'The rel error per event mean is: {rel_error.mean():.2f}, rel error per event std is: {rel_error.std():.2f}')
    output = df.output
    target = df.target
    print(f'Average number of particles per event:\n'
    f'output: {output.mean():.2f}, target: {target.mean():.2f}, rel error: {(output.mean()-target.mean())/target.mean()}')

def my_rel():
    my_path = os.path.join('.', 'csv_files', '1class_newtry')
    for i in np.linspace(10, 30, 3, dtype='int'):
        df = pd.read_csv(os.path.join(my_path, f'epoch_{i}', 'data_frame.csv'))
        plt.figure(figsize=(12, 6))
        plt.clf()
        plt.ylabel('relative error in %')
        plt.xlabel('target value')
        y = df.rel_error
        y *= 100
        x = df.target
        plt.scatter(x, y)
        plt.savefig(os.path.join(my_path, f'epoch_{i}', 'rel_error_fig.png'))


def rel_error_table():
    rel_mean_list = list()
    rel_std_list = list()
    epoch_list = list()
    my_path = os.path.join('csv_files', '2d_20classes', 'run_1')
    for i in np.linspace(10, 100, 10):
        if i.is_integer():
            i = int(i)
        print(f'Working on epoch_{i}')
        df = pd.read_csv(os.path.join(my_path, f'epoch_{i}', 'data_frame.csv'))
        rel_errors = df.rel_error * 100
        rel_mean = rel_errors.mean()
        rel_std = rel_errors.std()
        epoch_list.append(i)
        rel_mean_list.append(rel_mean)
        rel_std_list.append(rel_std)
    rel_df = pd.DataFrame(
    {'epoch': epoch_list,
     'mean[%]': rel_mean_list,
     'std[%]': rel_std_list
    })
    rel_df.to_csv(os.path.join(my_path, 'rel.csv'))

def rel_error_table_nonormal(folder_name, num_runs, num_epochs):
    
    num_runs = num_runs
    rel_mean_runs = list()
    rel_std_runs = list()
    for run in range(num_runs):
        rel_mean_list = list()
        rel_std_list = list()
        epoch_list = list()
        print(f'Working on run {run}')
        print('='*30)
        saved_path = folder_name
        my_path = os.path.join(f'{folder_name}', f'run_{run}')
        for i in np.linspace(10, num_epochs, int(num_epochs/10)):
            if i.is_integer():
                i = int(i)
            print(f'Working on epoch_{i}')
            df = pd.read_csv(os.path.join(my_path, f'epoch_{i}', 'data_frame.csv'))
            target = df.target
            output = df.output
            rel_mean = ((target - output)/1).mean()
            rel_std = ((target - output)/1).std()
            epoch_list.append(i)
            rel_mean_list.append(rel_mean)
            rel_std_list.append(rel_std)
        rel_mean_runs.append(rel_mean_list)
        rel_std_runs.append(rel_std_list)
    rel_mean_runs = np.array(rel_mean_runs)
    rel_mean_runs = np.stack(rel_mean_runs, axis=0).mean(axis=0)
    rel_std_runs = np.array(rel_std_runs)
    rel_std_runs = np.stack(rel_std_runs, axis=0).mean(axis=0)
    rel_df = pd.DataFrame(
    {'epoch': epoch_list,
    'mean': rel_mean_runs,
    'std': rel_std_runs
    })
    rel_df.to_csv(os.path.join(saved_path, 'rel_nonormal.csv'))
    plt.figure(num=0, figsize=(12, 6))
    plt.clf()
    plt.title('error as a function of epoch')
    plt.ylabel('rel error: target - output')
    plt.xlabel('epoch')
    plt.errorbar(epoch_list, rel_mean_runs, yerr=rel_std_runs)
    plt.savefig(os.path.join(saved_path, 'rel_nonormal'))
  
def show_noise():
    noise_file = os.path.join('./', 'data', 'raw', 'fast.elaser_randomised_bg')
    en_dep = loadmat(noise_file)['0']
    en_dep_noise = torch.zeros((110, 11, 21))
    for i in range(en_dep_noise.shape[0]):
        for j in range(en_dep_noise.shape[1]):
            for k in range(en_dep_noise.shape[2]):
                en_dep_noise[i,j,k] = en_dep[k,i,j]
    plt.figure(num=0, figsize=(12, 6))
    plt.clf()
    plt.imshow(en_dep_noise.sum(axis=1), interpolation="nearest", origin="upper", aspect="auto")
    plt.colorbar()
    plt.savefig('show_noise')
    return None

def excel_maker(folder_name, num_runs, num_classes):
    df_dict_output = dict()
    df_dict_target = dict()
    for run in range(num_runs):
        with open(os.path.join(folder_name,f'bin_results_run_{run}.txt'), 'r') as f:
            lines = f.readlines()
            output = lines[0].split('[')[1].split()[:num_classes]
            output[-1] = output[-1][:-1]
            # output = [float(x[:-2]) for x in output]
            output = [float(x[:-1]) for x in output]
            target = lines[0].split('[')[2].split()[:num_classes]
            target[-1] = target[-1][:-1]
            # target = [float(x[:-2]) for x in target]
            target = [float(x[:-1]) for x in target]
            df_dict_output[f'{run}'] = output
            df_dict_target[f'{run}'] = target
        df_output = pd.DataFrame(data=df_dict_output)
        df_target = pd.DataFrame(data=df_dict_target)
    df_output.to_csv(os.path.join('csv_files', 'output.csv'))
    df_target.to_csv(os.path.join('csv_files', 'target.csv'))
    return None

if __name__ == '__main__':
    # rely()

    # my_rel()
    # rel_error_table()

    # rel_error_table_nonormal()

    # show_noise()

    # rel_error_table_nonormal(folder_name='./csv_files/paper/3_to_5/5_micron', num_runs=1, num_epochs=110)
    excel_maker(folder_name='./saved/diff_run_res/3_to_5/5_micron', num_runs=1, num_classes=20)