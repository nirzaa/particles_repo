import os
# import matplotlib.pyplot as plt
import numpy as np
from shan_scripts import plots_1 as p1
from shan_scripts import plots as p

def rel_fig(num_case, epochs_every, total_epochs, total_runs, presentation=False, case=None):

    epochs_list = range(epochs_every, total_epochs+epochs_every, epochs_every)

    os.system("cat ./csv_files/multiple_runs/case_" + str(num_case) + "/stats.txt | grep 'relative error' | awk '{print $6}' | sed 's/\%//g' > ./csv_files/multiple_runs/case_" + str(num_case) + "/rel_error.txt")
    with open('./csv_files/multiple_runs/case_{}/rel_error.txt'.format(num_case)) as f:
        lines = f.readlines()

    cntr = 0
    losses = list()
    epochs_list = list(epochs_list) * total_runs
    for run in range(total_runs):
        for epoch in range(epochs_every, total_epochs+epochs_every, epochs_every):
            losses.append(float(lines[cntr].split('\n')[0].replace(',','')))
            cntr += 1
    if presentation:
        p.rel_error(epochs_list, losses, f'./shan_scripts/multiple_runs/case_{num_case}/rel_error.pdf')
        p.rel_error(epochs_list, losses, f'./shan_scripts/multiple_runs/case_{num_case}/rel_error.jpeg')
if __name__ == '__main__':
    pass




