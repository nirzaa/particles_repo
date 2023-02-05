import os
import re
import torch
import h5py
import numpy as np

def train_func():
    os.system('python3 ./train.py --config ./config.json')

def test_func(folder_name):
    files_list = os.listdir(f'./saved/models/new_model/{folder_name}')
    files_list.sort()
    files_list = files_list[:-2]
    LINE_RE = r'checkpoint-epoch(\d+).pth'

    for file in files_list:
        epoch_num = int(re.findall(LINE_RE, file)[0])
        print(f'Working on epoch num: {epoch_num}...')
        print('='*50)
        os.system(f"python3 ./test.py --resume ./saved/models/new_model/{folder_name}/checkpoint-epoch{epoch_num}.pth --c ./config.json")


if __name__ == '__main__':
    gpu_name = torch.cuda.get_device_name(0)
    print(f'We are using {gpu_name}')
    print('='*70)
    num_runs = 1
    for run in range(num_runs):
        with h5py.File(os.path.join('./', 'run_num.h5'), 'w') as f:
            dset = f.create_dataset("mydataset", data=run, dtype='int')
        # fix random seeds for reproducibility
        SEED = run
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        os.system('python3 ./utils/my_utils.py')
        print(f'This is the {run} run')
        print('='*50)
        os.system('rm ./saved/models/new_model/* -r')
        train_func()
        train_folder = os.listdir(f'./saved/models/new_model/')[0]
        test_func(folder_name=train_folder)
        os.system(f'mkdir ./csv_files/run_{run}')
        os.system(f'mv ./csv_files/epoch_* ./csv_files/run_{run}')
        # os.system('python3 ./analyze_auto.py')
