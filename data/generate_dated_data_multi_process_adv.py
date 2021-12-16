import os, math
import time

import numpy as np
import pandas as pd
import geopy.distance
import multiprocessing

from data.gcnm_utils import get_dist_matrix


'''
    Input:  "xxx.csv" with first column as "Date"
    Output: the saved files for preprocessed datasets, i.e., "train/val/test_dataTime.npz" including:
        - x
        - dateTime
        - y
        - max_speed
        - x_offsets
        - y_offsets

    Remarks: 
        - Centralised processing for generating local statistic features of traffic data
        - Costing! e.g., for PEMS-BAY, it takes more than four hours for 20% missing values
'''


def prepare_dataset(output_dit, df, x_offsets, y_offsets, masking, dists, L, S, mask_ones_proportion=0.8):
    """
        Prepare training & testing data integrating local statistic features
    :param output_dit: output path for saving
    :param df: (N, D), i.e., (num_samples, num_nodes)
    :param x_offsets: range(-11, 1)
    :param y_offsets: range(1, 13)
    :param masking:
    :param dists: the distance matrix (N, N) for the sensor nodes; directed or undirected
    :param L: the number of previous temporal measures to check
    :param S: the number of nearby spatial measures to check
    :param mask_ones_proportion:
    :return:
        x: (N, 8, L, D) including (x, Mask, X_last_obsv, X_mean_t, Delta_t, X_closest_obsv, X_mean_s, Delta_s)
        dateTime: (N, L)
        y: (N, L, D)
    """

    num_samples, num_nodes = df.shape
    data = df.values  # (num_samples, num_nodes)
    speed_tensor = data.clip(0, 100)  # (N, D)
    max_speed = speed_tensor.max().max()
    speed_tensor = speed_tensor / max_speed  # (N, D)

    date_array = df.index.values  # (N)
    print(speed_tensor.shape, date_array.shape)

    x, dateTime, y = [], [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = speed_tensor[t + x_offsets, ...]
        dateTime_t = date_array[t + x_offsets]
        y_t = speed_tensor[t + y_offsets, ...]
        x.append(x_t)
        dateTime.append(dateTime_t)
        y.append(y_t)
    speed_sequences = np.stack(x, axis=0)  # (N, L, D)
    dateTime = np.stack(dateTime, axis=0)  # (N, L)
    speed_labels = np.stack(y, axis=0)  # (N, L, D)

    # using zero-one mask to randomly set elements to zeros
    if masking:
        print('Split Speed/label finished. Start to generate Mask, Delta_t, Last_observed_X ...')
        np.random.seed(1024)
        Mask = np.random.choice([0, 1], size=(speed_sequences.shape),
                                p=[1 - mask_ones_proportion, mask_ones_proportion])  # (N, L, D)
        speed_sequences = np.multiply(speed_sequences, Mask)

        # temporal information -> to consider extracting the statistic feature from longer history data (caan probablement improve the performance)
        interval = 5  # 5 minutes
        s = np.zeros_like(speed_sequences)  # time stamps in (N, L, D)
        for i in range(s.shape[1]):
            s[:, i, :] = interval * i

        Delta_t = np.zeros_like(
            speed_sequences)  # time intervals, if all previous measures are missing, Delta_t[i, j, k] = 0, X_last_obsv[i, j ,k] = 0
        Delta_s = np.zeros_like(
            speed_sequences)  # spatial distance, if all variables are missing, Delta_s[i, j, k] = 0, X_closest_obsv[i, j ,k] = 0
        X_last_obsv = np.copy(speed_sequences)
        X_closest_obsv = np.copy(speed_sequences)
        X_mean_t = np.zeros_like(speed_sequences)
        X_mean_s = np.zeros_like(speed_sequences)

        for i in range(1, s.shape[1]):
            Delta_t[:, i, :] = s[:, i, :] - s[:, i - 1, :]  # calculate the exact minuites

        missing_index = np.where(Mask == 0)  # (array1, array2, array3), length of each array: number of missing values

        # X_mean_t, temporal mean for each segment
        start = time.time()
        for i in range(speed_sequences.shape[0]):  # N samples (N, L, D)
            for d in range(speed_sequences.shape[2]):
                temp_neighbor = speed_sequences[i, :, d]  # (L)
                nonzero_index = np.nonzero(temp_neighbor)  # return x arrays, for each we have xx elements
                if len(nonzero_index[0]) == 0:
                    continue
                else:
                    nonzero_temp_neighbor = temp_neighbor[nonzero_index]
                    avg = np.mean(nonzero_temp_neighbor, keepdims=True)
                    X_mean_t[i, :, d] = np.tile(avg, X_mean_t.shape[1])
        print("total time cost {}".format(time.time() - start))
        # save X_mean_t into ".npz" file
        X_mean_t_save_path = os.path.join(output_dit,
                                          "XMeanT_missRatio_{:.2f}%.npz".format((1 - mask_ones_proportion) * 100))
        np.savez_compressed(
            X_mean_t_save_path,
            X_mean_t=X_mean_t
        )
        print("X_mean_t is saved in ", X_mean_t_save_path)

        # spatial information
        dists_one_all_array = []
        sorted_node_ids_array = []
        for d in range(speed_sequences.shape[2]):
            dists_one_all = dists[d]  # the distance array between node k and all other nodes
            dists_one_all = list(enumerate(dists_one_all))  # [(idx, dist)]
            dists_one_all = sorted(dists_one_all, key=lambda x: x[1])  # by default ascending order
            sorted_node_ids = [x[0] for x in dists_one_all[:S]]  # only take the IDs of the S nearest nodes
            dists_one_all = {v[0]:v[1] for v in dists_one_all}  # {idx: dist}
            dists_one_all_array.append(dists_one_all)
            sorted_node_ids_array.append(sorted_node_ids)

        nbr_missing_all = missing_index[0].shape[0]
        nbr_finished = 0
        current_ratio = 0
        start = time.time()
        for idx in range(missing_index[0].shape[0]):  # number of missing values
            nbr_finished += 1
            finished_ratio = nbr_finished // (0.01 * nbr_missing_all)
            if finished_ratio != current_ratio:
                end = time.time()
                print("{}% of the (Delta_t, X_last_obsv) are calculated ! Accumulated time cost: {}s".format(
                    nbr_finished // (0.01 * nbr_missing_all), end - start))
                current_ratio = finished_ratio

            # index in (N, L, D)
            i = missing_index[0][idx]
            j = missing_index[1][idx]
            k = missing_index[2][idx]

            # Delta_t, X_last_obsv
            if j != 0 and j != min_t:  # if the missing value is in the middle of the sequence
                Delta_t[i, j + 1, k] = Delta_t[i, j + 1, k] + Delta_t[i, j, k]
            if j != 0:
                X_last_obsv[i, j, k] = X_last_obsv[
                    i, j - 1, k]  # last observation, can be zero, problem when handling long-range missing values
        print('Generate Delta_t, X_last_obsv finished \n')

        print('Start generating Delta_s, X_closest_obsv, X_mean_s \n')

        #extract missing value locations from "missing_index"
        # input: i, j, k; speed_sequencess[i, j, k]

        pool = multiprocessing.Pool()
        for idx in range(missing_index[0].shape[0]):  # number of missing values
            # Delta_s, X_closest_obsv
            i = missing_index[0][idx]
            j = missing_index[1][idx]
            k = missing_index[2][idx]
            dists_one_all = dists_one_all_array[k]  # {idx: dist}
            speeds = speed_sequences[i, j] # 1-d array: (D)
            sorted_node_ids = sorted_node_ids_array[k] # 1-d array: (D)

            for triple in dists_one_all:
                idx = triple[0]
                dist = triple[1]
                if speed_sequences[i, j, idx] != 0:
                    Delta_s[i, j, k] = dist
                    X_closest_obsv[i, j, k] = speeds[idx]
                    break
                else:
                    continue

            # X_mean_s
            spatial_neighbor = speeds[sorted_node_ids]  # S measures
            nonzero_index = np.nonzero(spatial_neighbor)  # return x arrays, for each we have xx elements
            if len(nonzero_index[0]) == 0:
                continue
            else:
                nonzero_spatial_neighbor = spatial_neighbor[nonzero_index]
                X_mean_s[i, j, k] = np.mean(nonzero_spatial_neighbor)

    print('Generate Mask, Last/Closest_observed_X, X_mean_t/s, Delta_t/s finished.')

    if masking:
        speed_sequences = np.expand_dims(speed_sequences, axis=1)
        Mask = np.expand_dims(Mask, axis=1)
        X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
        X_closest_obsv = np.expand_dims(X_closest_obsv, axis=1)
        X_mean_t = np.expand_dims(X_mean_t, axis=1)
        X_mean_s = np.expand_dims(X_mean_s, axis=1)
        Delta_t = np.expand_dims(Delta_t, axis=1)
        Delta_s = np.expand_dims(Delta_s, axis=1)
        dataset_agger = np.concatenate(
            (speed_sequences, Mask, X_last_obsv, X_mean_t, Delta_t, X_closest_obsv, X_mean_s, Delta_s),
            axis=1)  # (N, 8, L, D)

        return dataset_agger, dateTime, speed_labels, max_speed  # (N, 8, L, D), (N, L), (N, L, D)

    else:
        return speed_sequences, dateTime, speed_labels, max_speed  # (N, L, D), (N, L), (N, L, D)


def generate_train_val_test(traffic_df_filename, dist_filename, output_dir, masking, L, S,
                            train_val_test_split=[0.7, 0.1, 0.2], mask_ones_proportion=0.8):
    """
        To generate the splitted datasets
    :param traffic_df_filename:
    :param dist_file: distance matrix file
    :param output_dir: the path to save generated datasets
    :param masking: default True
    :param L: the recent sample numbers
    :param S: the nearby node numbers
    :param train_val_test_split: the splitting ratio
    :param mask_ones_proportion: the masking ratio
    :return:
        df: (N_all, D), the full dataframe including "dateTime" ass the first column
        save datasets into ".npz" files
        # x: (N, 8, L, D)
        # dateTime: (N, L)
        # y: (N, L, D)
    """
    df = pd.read_hdf(traffic_df_filename)
    df = df.iloc[:1000, :]
    sensor_locs = np.genfromtxt(dist_filename, delimiter=',')
    sensor_ids, sensor_id_to_ind, dist_mx = get_dist_matrix(sensor_locs)
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (N, 8, L, D)
    # dateTime: (N, L)
    # y: (N, L, D)
    x, dateTime, y, max_speed = prepare_dataset(
        output_dir,
        df,
        x_offsets,
        y_offsets,
        masking,
        dist_mx,
        L,
        S,
        mask_ones_proportion)
    print("x shape: ", x.shape, "dateTime shape: ", dateTime.shape, ", y shape: ", y.shape)

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_train = round(num_samples * train_val_test_split[0])
    num_test = round(num_samples * train_val_test_split[2])
    num_val = num_samples - num_test - num_train

    x_train, dateTime_train, y_train = x[:num_train], dateTime[:num_train], y[:num_train]
    x_val, dateTime_val, y_val = (
        x[num_train: num_train + num_val],
        dateTime[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, dateTime_test, y_test = x[-num_test:], dateTime[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _dateTime, _y = locals()["x_" + cat], locals()["dateTime_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        # x: (N, 8, L, D)
        # dateTime: (N, L)
        # y: (N, L, D)
        np.savez_compressed(
            os.path.join(output_dir, "%s_dateTime.npz" % cat),
            x=_x,
            dateTime=_dateTime,
            y=_y,
            max_speed=max_speed,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    return df


def retrieve_hist(dateTime, full_data, nh, nd, nw, tau):
    """

    :param dateTime: (B, L), numpy array
    :param full_data: (N, D) dataframe, with "dateTime" as the first column
    :return:
        A concatenated segment
    """

    B, L = dateTime.shape
    offsets = np.sort(np.arange(0, L, 1))
    full_data_index = full_data.index
    full_data_value = full_data.values
    Td = 12 * 24  # 12 measures per hour
    Tw = 12 * 24 * 7  # 12 measures per hour

    res_h, res_d, res_w = [], [], []
    if tau is None:
        tau = L

    for i in range(B):
        start_date, end_date = dateTime[i, 0], dateTime[i, L - 1]
        start, end = full_data_index.get_loc(start_date), full_data_index.get_loc(end_date)

        # recent observations
        start_h, end_h = start - nh * tau, end - L
        if start_h < 0:  # fill with current observation when no previous readings
            x = np.tile(full_data_value[start:end + 1], (nh * tau, 1, 1))  # (L, D) -> (nh*tau, L, D)
        else:
            x = []
            for t in range(start_h, end_h + 1):  # [start_h, end_h]
                x_t = full_data_value[t + offsets]  # (L, D)
                x.append(x_t)
            x = np.stack(x, axis=0)  # (nh*tau, L, D)
        res_h.append(x)

        # daily observations
        x_d = []
        curr_reading = np.tile(full_data_value[start:end + 1], (tau, 1, 1))  # (tau, L, D)
        for i in range(1, nd + 1, 1):
            start_d, end_d = start - i * Td - int(tau / 2), end - i * Td - L + int(tau / 2)
            if start_d < 0:
                x_d.append(curr_reading)
            else:
                x = []
                for t in range(start_d, end_d + 1):  # [start_d, end_d]
                    x_t = full_data_value[t + offsets]  # (L, D)
                    x.append(x_t)
                x = np.stack(x, axis=0)  # (tau, L, D)
                curr_reading = x
                x_d.append(x)
        x_d = np.concatenate(x_d, axis=0)  # (nd*tau, L, D)
        res_d.append(x_d)

        # weekly observations
        x_w = []
        curr_reading = np.tile(full_data_value[start:end + 1], (tau, 1, 1))  # (tau, L, D)
        for i in range(1, nw + 1, 1):
            start_w, end_w = start - i * Tw - int(tau / 2), end - i * Tw - L + int(tau / 2)
            if start_w < 0:
                x_w.append(curr_reading)
            else:
                x = []
                for t in range(start_w, end_w + 1):  # [start_d, end_d]
                    x_t = full_data_value[t + offsets]  # (L, D)
                    x.append(x_t)
                x = np.stack(x, axis=0)  # (tau, L, D)
                curr_reading = x
                x_w.append(x)
        x_w = np.concatenate(x_w, axis=0)  # (nw*tau, L, D)
        res_w.append(x_w)

    res_h = np.stack(res_h, axis=0)  # (B, nh*tau, L, D)
    res_d = np.stack(res_d, axis=0)  # (B, nd*tau, L, D)
    res_w = np.stack(res_w, axis=0)  # (B, nw*tau, L, D)

    return np.concatenate((res_h, res_d, res_w), axis=1)  # (B, nw*tau + nd*tau + nh *tau, L, D)


if __name__ == "__main__":
    root_path = "./Datasets/"
    datasets = ["PEMS/PEMS03/", "PEMS/PEMS04/", "PEMS/PEMS07/", "PEMS/PEMS08/", "PEMS-BAY/", "METR-LA/"]
    dataset = datasets[4]
    data_path = root_path + dataset  # "PEMS-BAY"

    traffic_df_filename = data_path + dataset[:-1].lower() + '.h5'  # raw_hdf file
    dist_filename = data_path + "graph_sensor_locations.csv"
    output_dir = data_path
    masking = True
    L = 12
    S = 5
    generate_train_val_test(traffic_df_filename, dist_filename, output_dir, masking, L, S,
                            train_val_test_split=[0.7, 0.1, 0.2], mask_ones_proportion=0.8)



