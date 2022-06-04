import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
# from torchsummary import summary


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        # print('Model output')
        # print('='*50)
        # # the shape is from data_loaders.py line 109
        # summary(self.model, (1, 110, 11, 21)) 
        
        self.train_metrics.reset()
        # for batch_idWx, (data, target) in enumerate(self.data_loader):
        for batch_idx, data in enumerate(self.data_loader):

            # data, target = data.to(self.device), target.to(self.device)
            # self.optimizer.zero_grad()
            # output = self.model(data)

            en_dep, target, _, _ = data

            # target = target / target.sum(axis=1).reshape(target.shape[0], 1)
            
            en_dep, target = en_dep.to(self.device), target.to(self.device)
            target = target.float()

            self.optimizer.zero_grad()
            output = self.model(en_dep)


            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        """
        Here I am preparing the tensors to accumulate the results of all of the batches.
        out and target are the model results and labels, and bias is the difference list between them.
        """
        tot_bias = torch.Tensor().to(self.device)
        tot_out = torch.Tensor().to(self.device)
        tot_target = torch.Tensor().to(self.device)

        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader):
                en_dep, target, _, _ = data
                en_dep, target = en_dep.to(self.device), target.to(self.device)
                target = target.float()
                output = self.model(en_dep)

                ####################################
                # Notice: sometimes its output and sometimes its output[:, 0], depends on some loss functions and formats.
                loss = self.criterion(output, target)
                bias = target - output
                ####################################

                # Accumulate the outputs, labels and biases.
                tot_bias = torch.cat((tot_bias, bias), 0)
                tot_target = torch.cat((tot_target, target), 0)
                tot_out = torch.cat((tot_out, output), 0)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        # Here we write the accumulated data.
        self.writer.add_histogram("tot_out", tot_out, bins='auto')
        self.writer.add_histogram("tot_bias", tot_bias, bins='auto')
        self.writer.add_histogram("tot_target", tot_target, bins='auto')

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            # current = batch_idx
            current = self.epoch
            # total = self.len_epoch
            total = self.epochs
        # return base.format(current, total, 100.0 * current / total)


        current = self.epoch
        total = self.epochs
        return base.format(current, total, 100.0 * current / total)
