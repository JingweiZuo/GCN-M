import os, json
import torch
import numpy as np

import argparse


class Exp_GCNMbasic(object):
    def __init__(self, config):
        self.config = config
        self.data_config = config['Data']
        self.model_config = config['Model']
        self.training_config = config['Training']

        # data config.
        self.root_path = self.data_config['root_path']
        self.data_path = self.data_config['data_path']
        self.dataset_name = self.data_config['dataset_name']
        self.data_split = json.loads(self.data_config['data_split'])  # load list
        self.dist_path = self.data_config['dist_path']
        self.adjdata = self.data_config['adjdata']
        self.adjtype = self.data_config['adjtype']
        self.mask_ones_proportion = float(self.data_config['mask_ones_proportion'])
        self.mask_option = self.data_config['mask_option']

        # model config
        self.model_name = self.model_config['model_name']
        self.in_dim = int(self.model_config['in_dim'])
        self.L = int(self.model_config['L'])
        self.S = int(self.model_config['S'])
        self.nh = int(self.model_config['nh'])
        self.nd = int(self.model_config['nd'])
        self.nw = int(self.model_config['nw'])
        self.tau = int(self.model_config['tau'])
        self.masking = json.loads(self.model_config['masking'].lower())
        self.pred_len = int(self.model_config['pred_len'])
        self.add_supports = json.loads(self.model_config['add_supports'].lower())

        # training config
        self.use_gpu = json.loads(self.training_config['use_gpu'].lower())
        self.gpu = int(self.training_config['gpu'])
        self.save_path = self.training_config['save_path']
        self.learning_rate = float(self.training_config['learning_rate'])
        self.lr_type = self.training_config['lr_type']
        self.patience = int(self.training_config['patience'])
        self.use_amp = json.loads(self.training_config['use_amp'].lower())
        self.batch_size =int(self.training_config['batch_size'])
        self.train_epochs = int(self.training_config['train_epochs'])

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)


    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            device = torch.device('cuda:{}'.format(self.gpu))
            print('Use GPU: cuda:{}'.format(self.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
