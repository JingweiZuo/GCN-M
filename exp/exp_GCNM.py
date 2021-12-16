import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from exp.exp_GCNMbasic import Exp_GCNMbasic
from data.gcnm_utils import load_dataset, get_dist_matrix, get_undirect_adjacency_matrix, load_adj
from data.generate_dated_data import generate_train_val_test, retrieve_hist
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, masked_mse, masked_mae, masked_mape
from models.model import STF_Informer, STF_InformerStack, DMSTGCN, gwnet, GCNM, GCNMdynamic

warnings.filterwarnings('ignore')


class Exp_GCNM(Exp_GCNMbasic):
    def __init__(self, config):
        super(Exp_GCNM, self).__init__(config) ## init device
        # result save
        testing_info = "model_{}_missR_{:.2f}_support_{}_{}".format(
            self.model_name,
            (1 - self.mask_ones_proportion) * 100,
            self.add_supports,
            self.dataset_name
            )
        self.save_path = self.save_path + testing_info + '/'

    def _build_model(self):
        #import different parameters for different models
        model_dict = {
            'STF_Informer': STF_Informer,
            'STF_InformerStack': STF_InformerStack,
            'DMSTGCN': DMSTGCN,
            'gwnet': gwnet,
            'GCNM': GCNM,
            'GCNMdynamic': GCNMdynamic
        }
        sensor_ids, sensor_id_to_ind, adj_mx = load_adj(self.adjdata, self.adjtype)
        if self.add_supports == True:
            supports = [torch.tensor(i).to(self.device) for i in adj_mx]
        else:
            supports = None
        model_name = self.model_name
        if model_name == 'GCNM' or model_name == 'GCNMdynamic':
            model = model_dict[model_name](
                self.device,
                num_nodes=adj_mx[0].shape[0],
                dropout=0.3,
                supports=supports,
                gcn_bool=True,
                addaptadj=True,
                aptinit=None,
                in_dim=self.in_dim,
                out_dim=12,
                residual_channels=32,
                dilation_channels=32,
                skip_channels=256,
                end_channels=512,
                kernel_size=2,
                blocks=4, layers=2
            )
        elif self.model_name == 'gwnet':
            model = model_dict[model_name](
                self.device,
                num_nodes=adj_mx[0].shape[0],
                dropout=0.3,
                supports=None,
                gcn_bool=True,
                addaptadj=True,
                aptinit=None,
                in_dim=self.in_dim,
                out_dim=12,
                residual_channels=32,
                dilation_channels=32,
                skip_channels=256,
                end_channels=512,
                kernel_size=2,
                blocks=4, layers=2
            )
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return model_optim

    def vali(self, vali_loader):
        self.model.eval()
        total_loss_mse = []
        total_loss_mae = []
        total_loss_mape = []
        for i, (batch_x, batch_dateTime, batch_y) in enumerate(vali_loader.get_iterator()):
            # batch_x: (B, 8, L, D)
            # batch_y: (B, L, D)

            if self.model_name == "gwnet":
                batch_x = batch_x[:, 0, ...]  # (B, 8, L, D) -> (B, L, D)
                batch_x = torch.Tensor(batch_x).to(self.device)
                batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, D)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)  # (B, L, D) -> (B, L, D)
                else:
                    outputs = self.model(batch_x)

            elif self.model_name == "GCNM" or self.model_name == "GCNMdynamic":
                x_hist = retrieve_hist(batch_dateTime, self.full_dataset, nh=self.nh, nd=self.nd, nw=self.nw,
                                       tau=self.tau)

                x_hist = torch.Tensor(x_hist).to(self.device)
                batch_x = torch.Tensor(batch_x).to(self.device)  # (B, 8, L, D)
                batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, D)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, x_hist)  # (B, 8, L, D), (B, n*tau, L, D) -> [N, L, D]
                else:
                    outputs = self.model(batch_x, x_hist)

            batch_y = batch_y[:, -self.pred_len:, :].to(self.device) #(N, L, D)

            pred = outputs.detach().cpu() # (B, L, D)
            true = batch_y.detach().cpu()
            pred = torch.mul(pred, torch.Tensor(self.max_speed))
            true = torch.mul(true, torch.Tensor(self.max_speed))

            loss_mse = masked_mse(pred, true, 0)
            loss_mae = masked_mae(pred, true, 0)
            loss_mape = masked_mape(pred, true, 0)

            total_loss_mse.append(loss_mse)
            total_loss_mae.append(loss_mae)
            total_loss_mape.append(loss_mape)
        total_loss_mse = np.average(total_loss_mse)
        total_loss_mae = np.average(total_loss_mae)
        total_loss_mape = np.average(total_loss_mape)
        self.model.train()
        # return total_loss
        return total_loss_mse, total_loss_mae, total_loss_mape

    def train(self):
        traffic_df_filename = os.path.join(self.root_path, self.data_path)
        dist_filename = os.path.join(self.root_path, self.dist_path) # the absolute distance matrix

        # full_dataset: (N, D)
        if self.mask_option == "random":
            stat_file = os.path.join(self.root_path, "random_missing", "randMissRatio_{:.2f}%.npz".format((1 - self.mask_ones_proportion) * 100))
        else:
            stat_file = os.path.join(self.root_path, "mix_missing",
                                     "mixMissRatio_{:.2f}%.npz".format((1 - self.mask_ones_proportion) * 100))
        '''generate_train_val_test(stat_file,
                                masking = self.masking,
                                train_val_test_split=self.data_split)
        '''
        # read the pre-processed & splitted dataset from file
        self.full_dataset, self.dataloader = load_dataset(traffic_df_filename, stat_file, self.batch_size, self.mask_ones_proportion)
        train_loader = self.dataloader['train_loader']
        vali_loader = self.dataloader['val_loader']
        test_loader = self.dataloader['test_loader']
        self.max_speed = self.dataloader['max_speed']
        print("self.max_speed is {}".format(self.max_speed))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        time_now = time.time()

        train_steps = train_loader.size
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        model_optim = self._select_optimizer()

        if self.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            train_loader.shuffle()
            for i, (batch_x, batch_dateTime, batch_y) in enumerate(train_loader.get_iterator()):

                iter_count += 1

                model_optim.zero_grad()
                # Query historical data
                # - (B, L), Dataframe (N, D) -> x_hist:(B, nw*tau + nd*tau + nh *tau, L, D)

                if self.model_name == "gwnet":
                    batch_x = batch_x[:, 0, ...] # (B, 8, L, D) -> (B, L, D)
                    batch_x = torch.Tensor(batch_x).to(self.device)
                    batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, D)

                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x)  # (B, L, D) -> (B, L, D)
                    else:
                        outputs = self.model(batch_x)

                elif self.model_name == "GCNM" or self.model_name == "GCNMdynamic":
                    x_hist = retrieve_hist(batch_dateTime, self.full_dataset, nh=self.nh, nd=self.nd, nw=self.nw, tau=self.tau)

                    x_hist = torch.Tensor(x_hist).to(self.device)
                    batch_x = torch.Tensor(batch_x).to(self.device)  #(B, 8, L, D)
                    batch_y = torch.Tensor(batch_y).to(self.device) #(B, L, D)

                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, x_hist) #(B, 8, L, D), (B, n*tau, L, D) -> [N, L, D]
                    else:
                        outputs = self.model(batch_x, x_hist)

                outputs = torch.mul(outputs, torch.Tensor(self.max_speed).to(self.device))# (B, L, D)
                batch_y = batch_y[:, -self.pred_len:, :].to(self.device) # (B, pred_len, D)
                batch_y = torch.mul(batch_y, torch.Tensor(self.max_speed).to(self.device))
                # loss = masked_mse(outputs, batch_y[:,:,:], 0)
                loss = masked_mae(outputs, batch_y, 0)  # [N, L, D]
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * (train_steps // self.batch_size) - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss_mse, vali_loss_mae, vali_loss_mape = self.vali(vali_loader)
            test_loss_mse, test_loss_mae, test_loss_mape = self.vali(test_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss mae: {2:.7f} Vali Loss mse: {3:.7f} Test Loss mse: {4:.7f} Vali mape: {5:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss_mse, test_loss_mse, vali_loss_mape))

            print("\n Test_mse: {0:.7f} Test_mae: {1:.7f} Test_mape: {2:.7f}".format(test_loss_mse, test_loss_mae,
                                                                                     test_loss_mape))
            early_stopping(vali_loss_mse, self.model, self.save_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.learning_rate, self.lr_type)

        best_model_path = self.save_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def test(self):
        test_loader = self.dataloader['test_loader']
        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_dateTime, batch_y) in enumerate(test_loader.get_iterator()):
            if self.model_name == "gwnet":
                batch_x = batch_x[:, 0, ...]  # (B, 8, L, D) -> (B, L, D)
                batch_x = torch.Tensor(batch_x).to(self.device)
                batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, D)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)  # (B, L, D) -> (B, L, D)
                else:
                    outputs = self.model(batch_x)

            elif self.model_name == "GCNM" or self.model_name == "GCNMdynamic":
                x_hist = retrieve_hist(batch_dateTime, self.full_dataset, nh=self.nh, nd=self.nd, nw=self.nw,
                                       tau=self.tau)  # (B, L), Dataframe (N, D) -> x_hist:(B, nw*tau + nd*tau + nh *tau, L, D)

                x_hist = torch.Tensor(x_hist).to(self.device)
                batch_x = torch.Tensor(batch_x).to(self.device)  # (B, 8, L, D)
                batch_y = torch.Tensor(batch_y).to(self.device)  # (B, L, D)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, x_hist)  # (B, 8, L, D), (B, n*tau, L, D) -> [N, L, D]
                else:
                    outputs = self.model(batch_x, x_hist)

            batch_y = batch_y[:, -self.pred_len:, :].to(self.device)  # (B, pred_len, D)
            pred = outputs.detach().cpu().numpy() * self.max_speed  # .squeeze()
            true = batch_y.detach().cpu().numpy() * self.max_speed # .squeeze()

            preds.append(pred)
            trues.append(true)

        #print('test shape 1:', preds.shape, trues.shape)
        preds = np.concatenate(preds, axis=0)   # [B, L, D] -> [N, L, D]
        trues = np.concatenate(trues, axis=0)  # [B, L, D] -> [N, L, D]
        print('test shape:', preds.shape, trues.shape)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('[Average value] mae:{}, rmse:{}, mape:{}'.format(mae, rmse, mape))

        np.save(self.save_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(self.save_path + 'pred.npy', preds)
        np.save(self.save_path + 'true.npy', trues)

        ## output the forecasting metric for each single point
        horizons = [int(self.pred_len * i / 12 - 1) for i in range(1, 13)]
        for horizon_i in horizons:
            mae, mse, rmse, mape, mspe = metric(preds[:, horizon_i, :], trues[:, horizon_i, :])
            print(
                "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}".format(
                    horizon_i + 1, mae, rmse, mape
                )
            )

        return
