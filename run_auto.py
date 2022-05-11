import os
import re
import torch
import h5py
import numpy as np
import sys
import random

def train_func():
    os.system('python ./train.py --config ./config.json')

def test_func(folder_name):
    files_list = os.listdir(f'./saved/models/new_model/{folder_name}')
    files_list.sort()
    files_list = files_list[:-2]
    LINE_RE = r'checkpoint-epoch(\d+).pth'

    for file in files_list:
        epoch_num = int(re.findall(LINE_RE, file)[0])
        print(f'Working on epoch num: {epoch_num}...')
        print('='*50)
        os.system(f"python ./test.py --resume ./saved/models/new_model/{folder_name}/checkpoint-epoch{epoch_num}.pth --c ./config.json")


if __name__ == '__main__':
    gpu_name = torch.cuda.get_device_name(0)
    print(f'We are using {gpu_name}')
    print('='*70)
    num_runs = 7
    with open('./run.txt', 'r') as f:
        run = int(f.read())
        print(f'Working on run = {run}')
    with h5py.File(os.path.join('./', 'run_num.h5'), 'w') as f:
        dset = f.create_dataset("mydataset", data=run, dtype='int')
    # fix random seeds for reproducibility
    SEED = run
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    os.system('python ./utils/my_utils.py')
    print(f'This is the {run} run')
    print('='*50)
    os.system('rm ./saved/models/new_model/* -r')
    train_func()
    train_folder = os.listdir(f'./saved/models/new_model/')[0]
    test_func(folder_name=train_folder)
    os.system(f'mkdir ./csv_files/run_{run}')
    os.system(f'mv ./csv_files/epoch_* ./csv_files/run_{run}')
    # os.system('python ./analyze_auto.py')

    folders = os.listdir('./csv_files')
    run_nums = [int(folder[4:]) for folder in folders if folder[:3] == 'run']
    if len(run_nums) >= 1:
        max_run = max(run_nums)
        if max_run < num_runs:
            with open('./run.txt', 'w') as f:
                f.write(f'{max_run+1}')
            python = sys.executable
            os.execl(python, python, * sys.argv)
        elif max_run == num_runs:
            print('We finished our run')
            with open('./run.txt', 'w') as f:
                f.write(f'0')
    
    
    elif len(run_nums) == 0:
        print('There is no runs in that folder')
    


