import os
import matplotlib.pyplot as plt

def calculate_loss(location, runs, epochs):

    if location.endswith('/'):
        location = location[:-1]

    for run in range(runs):
        plt.figure(figsize=(10,8))
        plt.clf()
        epoch_list = range(1, epochs+1)
        loss_train = list()
        loss_test = list()
        for epoch in epoch_list:
            loss_train.append(float(os.popen("grep -A 8 'Epoch: " + str(epoch) + " ' " + location + "/terminal_tmux_" + str(run) + ".txt | grep 'INFO:trainer:    loss' | awk -F':' '{print $4}'").read()))
            loss_test.append(float(os.popen("grep -A 8 'Epoch: " + str(epoch) + " ' " + location + "/terminal_tmux_" + str(run) + ".txt | grep 'INFO:trainer:    val_loss' | awk -F':' '{print $4}'").read()))
        plt.scatter(epoch_list, loss_train, label=f'run {run}: train')
        plt.scatter(epoch_list, loss_test, label=f'run {run}: test')
        # plt.ylim(0, 100)
        plt.ylim(0, 600)
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title("train loss vs. test loss")
        plt.legend()
        plt.savefig(f"{location}/run_{run}/loss.png")
        plt.savefig(f"{location}/loss_run_{run}.png")

if __name__ == '__main__':
    pass