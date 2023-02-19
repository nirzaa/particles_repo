import os
import matplotlib.pyplot as plt
import numpy as np



def rel_fig(num_case, epochs_every, total_epochs, total_runs):

    epochs_list = range(epochs_every, total_epochs+epochs_every, epochs_every)

    os.system("cat ./csv_files/multiple_runs/case_" + str(num_case) + "/stats.txt | grep 'relative error' | awk '{print $6}' | sed 's/\%//g' > ./csv_files/multiple_runs/case_" + str(num_case) + "/rel_error.txt")
    with open('./csv_files/multiple_runs/case_{}/rel_error.txt'.format(num_case)) as f:
        lines = f.readlines()

    cntr = 0
    plt.clf()
    for run in range(total_runs):
        losses = list()
        for epoch in range(epochs_every, total_epochs+epochs_every, epochs_every):
            losses.append(float(lines[cntr].split('\n')[0].replace(',','')))
            cntr += 1
        plt.scatter(epochs_list, losses, label=f'run {run}')
    plt.title(f'Case {num_case}')
    plt.ylim(-7.5, 7.5)
    plt.xlabel('Epochs')
    plt.ylabel('Relative error [%]')
    plt.legend(loc='center right', bbox_to_anchor=(1.1, 0.5))
    plt.savefig(f'./csv_files/multiple_runs/case_{num_case}/rel_error.jpeg')

if __name__ == '__main__':
    pass




