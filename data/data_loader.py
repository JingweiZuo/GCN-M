import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class StandardScaler():
    """
    Standard the input
    """

    def fit(self, data):
        self.mean = data.mean(0) #shape: (L, D) -> (D), the lasst dimension should match with s'D'
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean

class StandardScaler_Seduce():
    """
    Standard the input of SeduceCluster data
    """

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean[-2:]).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean[-2:]
        std = torch.from_numpy(self.std[-2:]).type_as(data).to(data.device) if torch.is_tensor(data) else self.std[-2:]
        return (data * std) + mean

class DynamicTrafficDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_split=[0.7, 0.1, 0.2],
                 data_path='ETTh1.csv',
                 timeenc=0, freq='T', days=288): #freq: the temporal indexing granularity - "minute"; days: 288*5mins = 24h
        # size [seq_len, label_len pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.ratio_train = data_split[0]
        self.ratio_test = data_split[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.timeenc = timeenc
        self.freq = freq
        self.days = days

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        df_raw = df_raw.fillna(0)
        num_train = int(len(df_raw) * self.ratio_train)
        num_test = int(len(df_raw) * self.ratio_test)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]  # (train_1, val_1, test_1)
        border2s = [num_train, num_train + num_vali, len(df_raw)]  # (train_2, val_2, test_2)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        #print('The columns of dataframe are ', df_raw.columns)
        cols_data = df_raw.columns[1:]  # without considering 'date' column
        df_data = df_raw[cols_data]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            # encode manually the temporal indexing feature
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            # what is the different here for 'self.timeenc'?
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        ## set indice of the temporal graph: the absolute position in raw sequence
        self.ind = np.arange(border1, border2)

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        self.data_x = self.scaler.transform(self.data_x)

    def __getitem__(self, index):
        ## to form x & y: (1, L, D)
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        ind = s_begin % self.days #output one single value for each sample

        return seq_x, seq_y, seq_x_mark, seq_y_mark, ind

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class TrafficDataset(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_split=[0.7, 0.1, 0.2],
                 data_path='x.csv',
                 timeenc=0, freq='T'): #freq: the temporal indexing granularity - "minute"
        # size [seq_len, label_len pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.ratio_train = data_split[0]
        self.ratio_test = data_split[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        df_raw = df_raw.fillna(0)
        num_train = int(len(df_raw) * self.ratio_train)
        num_test = int(len(df_raw) * self.ratio_test)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]  # (train_1, val_1, test_1)
        border2s = [num_train, num_train + num_vali, len(df_raw)]  # (train_2, val_2, test_2)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        #print('The columns of dataframe are ', df_raw.columns)
        cols_data = df_raw.columns[1:]  # without considering 'date' column
        df_data = df_raw[cols_data]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            # encode manually the temporal indexing feature
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['date'], 1).values #(L, 5)
        elif self.timeenc == 1:
            # what is the different here for 'self.timeenc'? Almost the same
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        train_data = df_data[border1s[0]:border2s[0]] #(total length, D)s
        self.scaler.fit(train_data.values)
        self.data_x = self.scaler.transform(self.data_x)

    def __getitem__(self, index):
        ## to form x & y: (1, L, D)
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end] #(L, 5)
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = np.expand_dims(seq_x, axis=-1)
        seq_y = np.expand_dims(seq_y, axis=-1)  #(L, D, 1)
        seq_x_mark = np.expand_dims(seq_x_mark, 1) #(L, 1, 5)
        seq_x_mark = np.tile(seq_x_mark, (1, seq_x.shape[1], 1)) #(L, D, 5)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_SeduceCluster(Dataset):
    # the server data from Seduce project: the data granularity is 1 second
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='train.csv',
                 target='Ti', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len pred_len]
        if size == None:
            self.seq_len = 10 * 60  # 10 mins
            self.label_len = 5 * 60  # 5 mins
            self.pred_len = 2 * 60  # 2 mins
        else:
            self.seq_len = size[0]  ## 600 steps
            self.label_len = size[1]  ## 300 steps
            self.pred_len = size[2]  ## 120 steps
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler_Seduce()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), sep=';')
        df_raw = df_raw.fillna(0)
        df_raw = df_raw.rename(columns={"Time": "date"})
        '''
        Keep the default input format for 'x' and the target variables ['Ti', 'To'] for 'y'
        '''
        Ti = df_raw.pop('Ti')
        To = df_raw.pop('To')
        df_raw.insert(7, 'Ti', Ti)
        df_raw.insert(8, 'To', To)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['second'] = df_stamp.date.apply(lambda row: row.second, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        if self.scale:  ## only transform data_x
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values) ## processing on GPU
            self.data_x = self.scaler.transform(self.data_x)

    def __getitem__(self, index):
        ## to form x & y
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        # the input data contains 2 variables for SeduceCluster
        # concatenant with the fake variables, "numpy object"
        N = data.shape[0]
        L = self.data_x.size()[-2]
        D = self.data_x.size()[-1]
        print("N is {}, L is {}, D is{}".format(N, L, D))
        fake_var = np.zeros(N, L, D)
        data = np.concatenate([fake_var, data], axis=-1)
        print("data.shape is {}".format(data.shape))
        return self.scaler.inverse_transform(data)[:, :, -2:]






