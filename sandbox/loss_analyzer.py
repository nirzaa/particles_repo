import os
import matplotlib.pyplot as plt

def calculate_loss(location, runs, epochs):

    if location.endswith('/'):
        location = location[:-1]

    plt.figure(figsize=(10,8))
    plt.clf()
    for run in range(runs):
        epoch_list = range(1, epochs+1)
        loss_train = list()
        loss_test = list()
        for epoch in epoch_list:
            loss_train.append(float(os.popen("grep 'Epoch: " + str(epoch) +" ' " + location + "/terminal_tmux_" + str(run) + ".txt | awk -F' ' '{print $7}'").read()))
            loss_test.append(float(os.popen("grep -A 25 'Working on epoch num: " + str(epoch) + "...$' " + location + "/terminal_tmux_" + str(run) + ".txt | grep 'INFO:test:{' | awk -F' ' '{print $2}'").read()[:6]))
        plt.scatter(epoch_list, loss_train, label=f'run {run}: train')
        plt.scatter(epoch_list, loss_test, label=f'run {run}: test')
    plt.ylim(0, 100)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title("train loss vs. test loss")
    plt.legend()
    plt.savefig(f"{location}/loss.png")

if __name__ == '__main__':
    pass