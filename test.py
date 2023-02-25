import argparse

import torch
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import my_utils
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.optimize import curve_fit

def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

NUM_CLASSES = 1
RAND_NUM = 0

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main(config):
    logger = config.get_logger('test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,  # 512
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    """
    Accumulation of batches - similar to training. Here we take the outputs, targets, bias and also the total number
    of showers in the batch and the samples indices in the files (was needed for different purposes).
    """
    tot_bias = torch.Tensor().to(device)
    tot_out = torch.Tensor().to(device)
    tot_target = torch.Tensor().to(device)
    tot_sums = torch.Tensor().to(device)
    tot_idx = torch.Tensor().to(device)

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            en_dep, target, num, idx = data

            ###############################################
            ### For testing on a totally random dataset ###
            # for i in range(RAND_NUM):
            #     en_dep[i] = torch.rand(torch.Size([1, 110, 11, 21]))
            ###############################################

            en_dep, target, num, idx = en_dep.to(device), target.to(device), num.to(device), idx.to(device)
            target = target.float()
            output = model(en_dep)
            

            my_target = target[:, 0]
            my_output = output[:, 0]

            my_target = target.sum(axis=1)
            my_output = output.sum(axis=1)
            
            my_rel_error = (my_target - my_output) / my_target            
            d = {'output': my_output.cpu(), 'target': my_target.cpu(), 'rel_error': my_rel_error.cpu()}
            df = pd.DataFrame(data=d, index=range(len(my_output)))
            flag = -1
            try: # 3-digit epoch
                epoch_num = int(str(config.resume)[-7:-4])
                flag = 0
            except:
                pass
            if flag == -1:
                try: # 2-digit epoch
                    epoch_num = int(str(config.resume)[-6:-4])
                    flag = 0
                except:
                    pass
            if flag == -1:
                try: # 1-digit epoch
                    epoch_num = int(str(config.resume)[-5:-4])
                    flag = 0
                except:
                    pass
            if flag == -1:
                epoch_num = 0
            my_path = os.path.join('./', 'csv_files', f'epoch_{epoch_num}')
            if not os.path.exists(my_path):
                os.makedirs(my_path)
            df = df.sort_values(by=['target'])
            df.to_csv(os.path.join(my_path, "data_frame.csv"))
            np.savetxt(os.path.join(my_path, 'events_numbers.csv'), idx.cpu(), delimiter=',')

            with h5py.File(os.path.join(my_path, 'data.h5'), 'w') as hf:
                hf.create_dataset('dataset_1', data=output.cpu())
                hf.create_dataset('dataset_2', data=target.cpu())
            plt.figure(num=0, figsize=(12, 6))
            plt.clf()
            plt.scatter(range(len(df['output'])), df['output'], label='output')
            plt.scatter(range(len(df['target'])), df['target'], label='target')
            plt.legend()
            plt.plot()
            chi_squared = ((df['output'] - df['target'])**2 / df['target']).mean()
            plt.title(f'Chi squared is: {chi_squared:.2f}')
            plt.savefig(os.path.join(my_path, 'output_target'))

            plt.figure(num=1, figsize=(12, 6))
            plt.clf()
            # y_value = df['rel_error']*100
            y_value = ((my_target - my_output) / my_target) * 100
            y_value = y_value.cpu()
            # y_value = y_value[y_value < 2]
            plt.scatter(range(len(y_value)), y_value, label='rel_error')
            
            parameters, covariance = curve_fit(Gauss, np.array(range(len(y_value))), y_value)
            plt.plot(range(len(y_value)), Gauss(np.array(range(len(y_value))), parameters[0], parameters[1]), label='Gauss distribution')
            plt.ylabel('rel error in %')
            plt.xlabel('events number')
            # plt.scatter(range(len(df['rel_error'])), df['rel_error'], label='rel_error')
            plt.legend()
            plt.plot()
            plt.title('rel error')
            plt.savefig(os.path.join(my_path, 'rel_error'))

            plt.figure(num=2, figsize=(12, 6))
            plt.clf()
            y_value = my_target - my_output
            # y_value = y_value[y_value < 2]
            plt.scatter(range(len(y_value)), y_value.cpu(), label='rel_error no normalization')
            
            parameters, covariance = curve_fit(Gauss, np.array(range(len(y_value))), y_value.cpu())
            plt.plot(range(len(y_value)), Gauss(np.array(range(len(y_value))), parameters[0], parameters[1]), label='Gauss distribution')
            plt.ylabel('rel error : (target - output)')
            plt.xlabel('events number')
            # plt.scatter(range(len(df['rel_error'])), df['rel_error'], label='rel_error')
            plt.legend()
            plt.plot()
            plt.title('rel error - no normalization')
            plt.savefig(os.path.join(my_path, 'rel_error_no_normal'))
            
            loss = loss_fn(output, target)
            bias = output - target

            # Accumulate the needed parameters
            tot_bias = torch.cat((tot_bias, bias), 0)
            tot_target = torch.cat((tot_target, target), 0)
            tot_out = torch.cat((tot_out, output), 0)
            tot_sums = torch.cat((tot_sums, num), 0)
            tot_idx = torch.cat((tot_idx, idx), 0)

            batch_size = en_dep.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    #################################################################################################
    ### My evaluation function for generating histograms, images and analysing the tests results ####
    my_utils.evaluate_test(tot_out.cpu().numpy(), tot_target.cpu().numpy(), tot_idx.cpu().numpy(),
                           tot_sums.cpu().numpy(), config)
    #################################################################################################

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
