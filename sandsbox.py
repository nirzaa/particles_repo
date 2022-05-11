import os
import sys
import numpy as np

np.random.seed(0)
print(np.random.rand())

num_runs = 3
with open('./run.txt', 'r') as f:
    run = int(f.read())
    print(f'Working on run = {run}')

folders = os.listdir('./csv_files')
run_nums = [int(folder[4:]) for folder in folders if folder[:3] == 'run']
if len(run_nums) >= 1:
    max_run = max(run_nums)
    if max_run < num_runs:
        with open('./run.txt', 'w') as f:
            f.write(f'{max_run+1}')
        python = sys.executable
        os.execl(python, python, * sys.argv)