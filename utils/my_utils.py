import fnmatch
import os
from collections import Counter
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
import h5py
import pandas as pd
import random
# from scipy.optimize import curve_fit

my_path = os.path.join('./')
sys.path.append(my_path)
from data_loader.data_loaders import Bin_energy_data



project_path = Path(__file__).parent.parent
fig_path = os.path.join('./', 'saved', 'figs')
res_path = os.path.join('./', 'saved', 'diff_run_res')

def merge_and_split_data(path, relation, moment, min_shower_num, max_shower_num, file):
    """
    Merge the raw data files into concatenated dataset then split by the relation to train and test set and save in the
     folders.
    Notice that the train is split into train\valid sets in the train function later.
    """
    dl = []
    for i in file:
        # edep_file = path / "raw" / f"signal.al.elaser.IP0{i}.edeplist.mat"
        # en_file = path / "raw" / f"signal.al.elaser.IP0{i}.energy.mat"

        edep_file = os.path.join('./', 'data', 'raw', f'signal.al.elaser.IP0{i}.edeplist.mat')
        edep_file_noise = os.path.join('./', 'data', 'raw', 'fast.elaser_randomised_bg')
        en_file = os.path.join('./', 'data', 'raw', f'signal.al.elaser.IP0{i}.energy.mat')

        dataset = Bin_energy_data(edep_file, en_file, moment=moment,
                                  min_shower_num=min_shower_num, max_shower_num=max_shower_num, file=i, noise_file=edep_file_noise)
        dl.append(dataset)

    dataset = torch.utils.data.ConcatDataset(dl)

    # mean, std = get_mean_and_std(dataset)
    # print(mean, std)

    train_d, test_d = torch.utils.data.random_split(dataset, [int(relation * len(dataset))+2,
                                                              int((1 - relation) * len(dataset))-1])

    # torch.save(train_d, path / "train//train.pt")
    # torch.save(test_d, path / "test//test.pt")

    torch.save(train_d, os.path.join('./', 'data', 'train', 'train.pt'))
    torch.save(test_d, os.path.join('./', 'data', 'test', 'test.pt'))


def test_bins(output, target, nums, bin_num=10, name=None, run_num='0', config=0):
    """Analysis of the bin results from test"""

    # Generate the bins

    hf = h5py.File(os.path.join('./', 'num_classes.h5'), 'r')
    num_classes = hf.get('dataset_1')
    num_classes = int(np.array(num_classes))
    hf.close()

    bin_num = num_classes

    total_out = [0] * bin_num
    total_target = [0] * bin_num
    bars = np.linspace(0, 13, bin_num)

    # Sum over al of the bin results for each bin.
    for i in range(bin_num):
        out_sum = sum(t[i] for t in output)
        target_sum = sum(t[i] for t in target)
        total_out[i] = out_sum
        total_target[i] = target_sum
    

    # Calculate the entropy of the bin lists for output and the truelabel bins.

    out_entropy = -sum([f * np.log(f) if f > 0 else 0 for f in total_out])
    tar_entropy = -sum([f * np.log(f) if f > 0 else 0 for f in total_target])
    b = out_entropy - tar_entropy

    print(f'Entopies: True distribution: {tar_entropy:.3f}, Predicted distribution: {out_entropy:.3f}, Bias: {b:.3f}')

    KL_1 = sum(
        [f * np.log((f + 0.0000001) / (total_target[i] + 0.0000001)) for i, f in enumerate(total_out)])
    KL_2 = sum(
        [f * np.log((f + 0.0000001) / (total_out[i] + 0.0000001)) for i, f in enumerate(total_target)])
    D = KL_1 + KL_2

    print(f'KL dist - KL1(q=target): {KL_1:.3f}, KL2(q=output): {KL_2:.3f}, Symmetric: {D:.3f}')
    print(f'total out: {["%.3f" % item for item in total_out]}')
    print(f'total target: {["%.3f" % item for item in total_target]}')
    print(f'total N: {sum(total_out)}, target N: {sum(total_target)}')
    bars = [float(f'{i:.2f}') for i in bars]

    # Text generation for bin legend
    text = 'Bin Energy range [GeV]: \n'
    for i in range(bin_num - 1):
        text += f'{i}: {bars[i]:.1f} - {bars[i + 1]:.1f} \n'
    # print(text)

    tot_mean_en_pred = []
    tot_mean_en_true = []
    for i in range(bin_num - 1):
        me = bars[i] + 0.35
        e_p = me * total_out[i]
        e_t = me * total_target[i]
        tot_mean_en_pred.append(e_p)
        tot_mean_en_true.append(e_t)

    mean_en_p = sum(tot_mean_en_pred) / sum(total_out)
    mean_en_t = sum(tot_mean_en_true) / sum(total_target)

    print(f'Mean E: {mean_en_p}, target E: {mean_en_t}')

    rng = [i + 1 for i in range(num_classes)]

    plt.figure(figsize=(28,20))
    plt.bar(rng, total_out, label='output', alpha=0.5)
    # plt.errorbar(rng, total_out, yerr=(1 / np.sqrt(np.abs(total_out))), fmt="+", color="b")
    plt.bar(rng, total_target, label='true_val', alpha=0.3)
    plt.xlabel('bins number for energies')
    plt.ylabel('number of particles')
    # plt.text(15.5, 0.015, text,
    plt.text(15.5, 0.015, text,
             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3}, fontsize='x-small')

    plt.xticks(rng, rotation=65)
    plt.title(f'{len(output)} samples')
    plt.legend()
    plt.savefig(os.path.join(res_path, f'binsgraph_run_{run_num}.png'))
    flag = -1
    try:
        epoch_num = int(str(config.resume)[-7:-4])
        flag = 0
    except:
        pass
    if flag == -1:
        try:
            epoch_num = int(str(config.resume)[-6:-4])
            flag = 0
        except:
            pass
    if flag == -1:
        epoch_num = 0
    csv_path = os.path.join('./csv_files', f'epoch_{epoch_num}')
    plt.savefig(os.path.join(csv_path, f'binsgraph_run_{run_num}.png'))

    np.savetxt(os.path.join(csv_path, 'hist_output.csv'), output, delimiter=',')
    np.savetxt(os.path.join(csv_path, 'hist_target.csv'), target, delimiter=',')

    # plt.show()
    plt.clf()
    save = {'output bins': total_out, 'output entropy': out_entropy,
            'target bins': total_target, 'target entropy': tar_entropy,
            'KL': KL_2}

    res_file = open(os.path.join(res_path, f'bin_results_run_{run_num}.txt'), 'wt')
    res_file.write(str(save))
    res_file.close()
    return

def evaluate_test(output, target, incdices, shower_nums, config):
    """
        This function evaluates the test results. The first part relates to the N prediction - assuming we have 1 class.
        The second part handles the 20 bin prediction.
    """
    with h5py.File(os.path.join('./', 'run_num.h5'), 'r') as f:
        run_num = int(np.array(f.get('mydataset')))
    file_tag = str(run_num)

    # evaluate_xy(output=output[:, 0], target=target[:, 0], run_num=file_tag)
    # np.savetxt(res_path / f'output_{file_tag}.txt', output[:, 0], delimiter="\n", fmt='%1.2f')


    # For each sample evaluate N and compare with Ntrue - get Nbias. Produce halina graphs with them and our graphs:
    # N_pred = np.sum(output, axis=1)  # Total number of showers in the test set
    # H_graphs(N_true=shower_nums, N_pred=N_pred, run_num=file_tag)  # Generate relevant graph images

    # Bins - sum over output bins and target bins. Compare graphs. produce PNG. Save the bins as text to calculate
    test_bins(output, target, shower_nums, bin_num=20, run_num=file_tag, config=config)

    ################################################################
    ######### Save idx list as txt file with the file tag ##########
    # np.savetxt(res_path / f'idx_run_{file_tag}.csv', [int(i) for i in incdices], delimiter=",", fmt='%i')
    ################################################################

    return

if __name__ == '__main__':
    # EDA("C:\\Users\\elihu\\PycharmProjects\\LUXE\\nongitdata\\Multiple Energies\\")
    # data_path = Path("C:\\Users\\elihu\\PycharmProjects\\LUXE\\LUXE-project-master\\data\\")
    with open('./run.txt', 'r') as f:
        run = int(f.read())
    SEED = run
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    data_path = Path(my_path = os.path.join('./', 'data'))
    merge_and_split_data(data_path, 0.8, moment=3, min_shower_num=1, max_shower_num=50000, file=[5])
    exit()