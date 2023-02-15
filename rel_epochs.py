import os
import matplotlib.pyplot as plt
import numpy as np

num_case = 1
epochs_every = 5
total_epochs = 40
total_runs = 10

epochs_list = range(epochs_every, total_epochs+epochs_every, epochs_every)

os.system("cat ./csv_files/multiple_runs/case_" + str(num_case) + "/stats.txt | grep 'relative error' | awk '{print $6}' | sed 's/\%//g' > ./csv_files/multiple_runs/case_" + str(num_case) + "/loss.txt")
with open('./csv_files/multiple_runs/case_{}/loss.txt'.format(num_case)) as f:
    lines = f.readlines()

cntr = 0
plt.clf()
for run in range(total_runs):
    losses = list()
    for epoch in range(epochs_every, total_epochs+epochs_every, epochs_every):
        losses.append(float(lines[cntr].split('\n')[0]))
        cntr += 1
    plt.scatter(epochs_list, losses, label=f'run {run}')
plt.title(f'Case {num_case}')
plt.ylim(-7.5, 7.5)
plt.xlabel('Epochs')
plt.ylabel('Relative error [%]')
plt.legend(loc='center right', bbox_to_anchor=(1.1, 0.5))
plt.savefig(f'./csv_files/multiple_runs/case_{num_case}/loss.jpeg')





