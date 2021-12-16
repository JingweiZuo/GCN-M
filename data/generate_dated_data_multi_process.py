import os, math
import time

import numpy as np
import pandas as pd
import geopy.distance
from concurrent.futures import ProcessPoolExecutor

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
        - Multi-process processing for generating local statistic features of traffic data
        - Costing! e.g., for PEMS-BAY, it takes more than two hours for 20% missing values
'''


def get_dist_matrix(sensor_locs):
    """
    Compute the absolute spatial distance matrix

    :param sensor_locs: with header and index, [index, sensor_id, longitude, latitude]
    :return:
    """
    sensor_ids = sensor_locs[1:, 1]  # remove header and index
    sensor_id_to_ind = {}
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind.update({sensor_id: i})
    for id1 in sensor_ids:
        coords_1 = sensor_locs[sensor_locs[:, 1] == id1][0][2:]
        for id2 in sensor_ids:
            if math.isinf(dist_mx[sensor_id_to_ind[id1], sensor_id_to_ind[id2]]):
                coords_2 = sensor_locs[sensor_locs[:, 1] == id2][0][2:]
                dist = round(geopy.distance.distance(coords_1, coords_2).km, 2)
                dist_mx[sensor_id_to_ind[id1], sensor_id_to_ind[id2]] = dist
                dist_mx[sensor_id_to_ind[id2], sensor_id_to_ind[id1]] = dist
            else:
                continue
    return sensor_ids, sensor_id_to_ind, dist_mx


def cal_statistics(idx):  # index in (N, L, D)
    global speed_sequences
    global Delta_t, X_last_obsv, Delta_s, X_closest_obsv, X_mean_t, X_mean_s, missing_index
    global dists_one_all_array, sorted_node_ids_array

    i = missing_index[0][idx]
    j = missing_index[1][idx]
    k = missing_index[2][idx]

    # Delta_t, X_last_obsv
    ## ******* May cause problem when computing Delta_t due to the random computing order.
    if j != 0 and j != speed_sequences.shape[1] - 1:  # if the missing value is in the middle of the sequence
        Delta_t[i, j + 1, k] = Delta_t[i, j + 1, k] + Delta_t[i, j, k]
    if j != 0:
        X_last_obsv[i, j, k] = X_last_obsv[
            i, j - 1, k]  # last observation, can be zero, problem when handling long-range missing values

    # Delta_s, X_closest_obsv
    dists_one_all = dists_one_all_array[k]  # [(idx, dist)]
    for triple in dists_one_all:
        idx = triple[0]
        dist = triple[1]
        if speed_sequences[i, j, idx] != 0:
            Delta_s[i, j, k] = dist
            X_closest_obsv[i, j, k] = speed_sequences[i, j, idx]
            break
        else:
            continue

    # X_mean_s
    sorted_node_ids = sorted_node_ids_array[k]
    spatial_neighbor = speed_sequences[i, j, sorted_node_ids]  # S measures
    nonzero_index = np.nonzero(spatial_neighbor)  # return x arrays, for each we have xx elements
    if len(nonzero_index[0]) != 0:
        nonzero_spatial_neighbor = spatial_neighbor[nonzero_index]
        X_mean_s[i, j, k] = np.mean(nonzero_spatial_neighbor)
    return idx



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

    global speed_sequences
    global Delta_t, X_last_obsv, Delta_s, X_closest_obsv, X_mean_t, X_mean_s, missing_index
    global dists_one_all_array, sorted_node_ids_array

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
        nbr_all = speed_sequences.shape[0] * speed_sequences.shape[2]
        nbr_finished = 0
        current_ratio = 0
        for i in range(speed_sequences.shape[0]):  # N samples
            for d in range(speed_sequences.shape[2]):
                nbr_finished += 1
                finished_ratio = nbr_finished // (0.01 * nbr_all)
                if finished_ratio != current_ratio:
                    print("{}% of X_mean_t are calculated ! Accumulated time cost: {}s" \
                        .format(nbr_finished // (0.01 * nbr_all), time.time() - start))
                    current_ratio = finished_ratio
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
            sorted_node_ids = [x[0] for x in dists_one_all[:S]]  # only take S nearest nodes

            dists_one_all_array.append(dists_one_all)
            sorted_node_ids_array.append(sorted_node_ids)

        executor = ProcessPoolExecutor()  # default, using nbr_CPU processes
        idx_miss = list(range(missing_index[0].shape[0]))
        nbr_all = len(idx_miss)
        nbr_temp = 0
        start = time.time()
        current_ratio = 0
        for res in executor.map(cal_statistics, idx_miss):
            nbr_temp += 1
            finished_ratio = nbr_temp // (0.01 * nbr_all)
            if finished_ratio != current_ratio:
                end = time.time()
                print("{} % of the statistic features are calculated ! Time cost: {}s".format(
                    nbr_temp // (0.01 * nbr_all), end - start))
                current_ratio = finished_ratio

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
        mask_ones_proportion
    )
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
            os.path.join(output_dir, "{}_missRatio_{:.2f}%_dateTime.npz".format(cat, (1 - mask_ones_proportion) * 100)),
            x=_x,
            dateTime=_dateTime,
            y=_y,
            max_speed=max_speed,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )

    return df


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

