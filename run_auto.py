import os
import re
import torch
import h5py
import numpy as np
import time

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
    # python -u run_auto.py | tee ./csv_files/terminal.txt
    os.system('clear')
    os.system('tmux clear-history')
    gpu_name = torch.cuda.get_device_name(0)
    print(f'We are using {gpu_name}')
    print('='*70)
    num_runs = 10
    os.system('tmux capture-pane -pS - > ./csv_files/terminal_tmux.txt')
    os.system('clear')
    os.system('tmux clear-history')
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
        os.system('tmux capture-pane -pS - >> ./csv_files/terminal_tmux.txt')
        os.system('clear')
        os.system('tmux clear-history')
        train_func()
        os.system('tmux capture-pane -pS - >> ./csv_files/terminal_tmux.txt')
        os.system('clear')
        os.system('tmux clear-history')
        train_folder = os.listdir(f'./saved/models/new_model/')[0]
        test_func(folder_name=train_folder)
        os.system('tmux capture-pane -pS - >> ./csv_files/terminal_tmux.txt')
        os.system('clear')
        os.system('tmux clear-history')
        os.system(f'mkdir ./saved/models/saved_models/run_{run}')
        os.system(f'mv ./saved/models/new_model/* ./saved/models/saved_models/run_{run}/')

        os.system(f'mkdir ./csv_files/run_{run}')
        os.system(f'mv ./csv_files/epoch_* ./csv_files/run_{run}')

        os.system('tmux capture-pane -pS - >> ./csv_files/terminal_tmux.txt')
        os.system('clear')
        os.system('tmux clear-history')
        # os.system('python3 ./analyze_auto.py')
    # os.system('tmux capture-pane -pS - >> ./csv_files/terminal_tmux.txt') # https://burnicki.pl/en/2021/07/04/dump-tmux-pane-history-to-a-file.html
    os.system('tmux capture-pane -pS - >> ./csv_files/terminal_tmux.txt')
    os.system('clear')
    os.system('tmux clear-history')