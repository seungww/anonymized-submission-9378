import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, Binarizer, MaxAbsScaler,
    Normalizer, PowerTransformer, QuantileTransformer, RobustScaler
)


def normalize_data(train_data, val_data, test_data, norms):
    normalized_train, normalized_val, normalized_test = [], [], []
    scalers, token_dicts = {}, {}

    for i, norm in enumerate(norms):
        train_value = train_data[i]
        val_value = val_data[i]
        test_value = test_data[i]

        if norm != 'token':
            # Convert all values to numeric safely
            train_value = np.nan_to_num(pd.to_numeric(train_value.flatten(), errors='coerce'), nan=0).reshape(train_value.shape)
            val_value = np.nan_to_num(pd.to_numeric(val_value.flatten(), errors='coerce'), nan=0).reshape(val_value.shape)
            test_value = np.nan_to_num(pd.to_numeric(test_value.flatten(), errors='coerce'), nan=0).reshape(test_value.shape)

        if norm in ['minmax', 'zscore', 'binary', 'maxabs', 'l1norm', 'l2norm', 'power', 'quantile', 'robust']:
            scaler = {
                'minmax': MinMaxScaler(),
                'zscore': StandardScaler(),
                'binary': Binarizer(),
                'maxabs': MaxAbsScaler(),
                'l1norm': Normalizer(norm='l1'),
                'l2norm': Normalizer(norm='l2'),
                'power': PowerTransformer(),
                'quantile': QuantileTransformer(),
                'robust': RobustScaler()
            }[norm]

            train_flat = train_value.flatten().reshape(-1, 1)
            train_value = scaler.fit_transform(train_flat).reshape(train_value.shape)
            val_value = scaler.transform(val_value.flatten().reshape(-1, 1)).reshape(val_value.shape)
            test_value = scaler.transform(test_value.flatten().reshape(-1, 1)).reshape(test_value.shape)

            scalers[i] = scaler

        elif norm == 'token':
            train_value = np.where(pd.isna(train_value), '<NA>', train_value)
            unique_values = np.unique(train_value)
            unique_values = unique_values[unique_values != "<NA>"]
            token_dict = {val: idx + 1 for idx, val in enumerate(unique_values)}
            token_dict['<NA>'] = 0
            token_dicts[i] = token_dict

            train_value = np.vectorize(lambda x: token_dict.get(x, 0))(train_value)
            val_value = np.vectorize(lambda x: token_dict.get(x, 0))(val_value)
            test_value = np.vectorize(lambda x: token_dict.get(x, 0))(test_value)

        normalized_train.append(train_value)
        normalized_val.append(val_value)
        normalized_test.append(test_value)

        logging.info(f"Feature {i} normalized using {norm}")
        logging.info(f"Train shape: {train_value.shape}, Val shape: {val_value.shape}, Test shape: {test_value.shape}")

    return normalized_train, normalized_val, normalized_test, scalers, token_dicts

def normalize_data_for_longitudinal_evaluation(train_data, val_data, test_1_data, test_2_data, test_3_data, norms):
    normalized_train, normalized_val, normalized_test_1, normalized_test_2, normalized_test_3 = [], [], [], [], []
    scalers = {}  # Dictionary to store scalers per feature
    token_dicts = {}  # Dictionary to store token mappings

    for i, norm in enumerate(norms):
        train_value = train_data[i]
        val_value = val_data[i]
        test_1_value = test_1_data[i]
        test_2_value = test_2_data[i]
        test_3_value = test_3_data[i]

        if norm != 'token':
            train_value = np.nan_to_num(pd.to_numeric(train_value.flatten(), errors='coerce'), nan=0).reshape(train_value.shape)
            val_value = np.nan_to_num(pd.to_numeric(val_value.flatten(), errors='coerce'), nan=0).reshape(val_value.shape)
            test_1_value = np.nan_to_num(pd.to_numeric(test_1_value.flatten(), errors='coerce'), nan=0).reshape(test_1_value.shape)
            test_2_value = np.nan_to_num(pd.to_numeric(test_2_value.flatten(), errors='coerce'), nan=0).reshape(test_2_value.shape)
            test_3_value = np.nan_to_num(pd.to_numeric(test_3_value.flatten(), errors='coerce'), nan=0).reshape(test_3_value.shape)

        if norm in ['minmax', 'zscore', 'binary', 'maxabs', 'l1norm', 'l2norm', 'power', 'quantile', 'robust']:
            scaler = {
                'minmax': MinMaxScaler(),
                'zscore': StandardScaler(),
                'binary': Binarizer(),
                'maxabs': MaxAbsScaler(),
                'l1norm': Normalizer(norm='l1'),
                'l2norm': Normalizer(norm='l2'),
                'power': PowerTransformer(),
                'quantile': QuantileTransformer(),
                'robust': RobustScaler()
            }[norm]

            train_flat = train_value.flatten().reshape(-1, 1)
            train_value = scaler.fit_transform(train_flat).reshape(train_value.shape)
            val_value = scaler.transform(val_value.flatten().reshape(-1, 1)).reshape(val_value.shape)
            test_1_value = scaler.transform(test_1_value.flatten().reshape(-1, 1)).reshape(test_1_value.shape)
            test_2_value = scaler.transform(test_2_value.flatten().reshape(-1, 1)).reshape(test_2_value.shape)
            test_3_value = scaler.transform(test_3_value.flatten().reshape(-1, 1)).reshape(test_3_value.shape)

            scalers[i] = scaler

        elif norm == 'token':
            train_value = np.where(pd.isna(train_value), '<NA>', train_value)
            unique_values = np.unique(train_value)
            unique_values = unique_values[unique_values != "<NA>"]
            token_dict = {val: idx + 1 for idx, val in enumerate(unique_values)}
            token_dict['<NA>'] = 0
            token_dicts[i] = token_dict

            train_value = np.vectorize(lambda x: token_dict.get(x, 0))(train_value)
            val_value = np.vectorize(lambda x: token_dict.get(x, 0))(val_value)
            test_1_value = np.vectorize(lambda x: token_dict.get(x, 0))(test_1_value)
            test_2_value = np.vectorize(lambda x: token_dict.get(x, 0))(test_2_value)
            test_3_value = np.vectorize(lambda x: token_dict.get(x, 0))(test_3_value)

        normalized_train.append(train_value)
        normalized_val.append(val_value)
        normalized_test_1.append(test_1_value)
        normalized_test_2.append(test_2_value)
        normalized_test_3.append(test_3_value)

        logging.info(f"Feature {i} normalized using {norm}")
        logging.info(f"Train shape: {train_value.shape}, Val shape: {val_value.shape}, Test1 shape {test_1_value.shape}, Test2 shape {test_2_value.shape}, Test3 shape {test_3_value.shape}")

    return normalized_train, normalized_val, normalized_test_1, normalized_test_2, normalized_test_3, scalers, token_dicts

def load_batch(norms, train_data, val_data, test_data, batch_size):
    x_train_list, y_train = train_data
    x_val_list, y_val = val_data
    x_test_list, y_test = test_data

    x_train_list, x_val_list, x_test_list, scalers, token_dicts = normalize_data(
        x_train_list, x_val_list, x_test_list, norms
    )

    def to_tensor(data_list, norms):
        tensors = []
        for x, norm in zip(data_list, norms):
            if norm == 'token':
                tensor = torch.tensor(x, dtype=torch.long)
            else:
                tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
            tensors.append(tensor)
        return tensors

    x_train_tensors = to_tensor(x_train_list, norms)
    x_val_tensors = to_tensor(x_val_list, norms)
    x_test_tensors = to_tensor(x_test_list, norms)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(*x_train_tensors, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*x_val_tensors, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(*x_test_tensors, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, scalers, token_dicts

def load_batch_for_longitudinal_evaluation(norms, train, val, t1, t2, t3, batch_size):
    xt, yt = train
    xv, yv = val
    xt1, yt1 = t1
    xt2, yt2 = t2
    xt3, yt3 = t3

    xt, xv, xt1, xt2, xt3, scalers, token_dicts = normalize_data_for_longitudinal_evaluation(
        xt, xv, xt1, xt2, xt3, norms
    )

    def to_tensor(data_list, norms):
        tensors = []
        for x, norm in zip(data_list, norms):
            if norm == 'token':
                tensor = torch.tensor(x, dtype=torch.long)  # shape: (B, T)
            else:
                tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # shape: (B, T, 1)
            tensors.append(tensor)
        return tensors

    xt = to_tensor(xt, norms)
    xv = to_tensor(xv, norms)
    xt1 = to_tensor(xt1, norms)
    xt2 = to_tensor(xt2, norms)
    xt3 = to_tensor(xt3, norms)

    yt = torch.tensor(yt, dtype=torch.long)
    yv = torch.tensor(yv, dtype=torch.long)
    yt1 = torch.tensor(yt1, dtype=torch.long)
    yt2 = torch.tensor(yt2, dtype=torch.long)
    yt3 = torch.tensor(yt3, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(*xt, yt), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*xv, yv), batch_size=batch_size)
    t1_loader = DataLoader(TensorDataset(*xt1, yt1), batch_size=batch_size)
    t2_loader = DataLoader(TensorDataset(*xt2, yt2), batch_size=batch_size)
    t3_loader = DataLoader(TensorDataset(*xt3, yt3), batch_size=batch_size)

    return train_loader, val_loader, t1_loader, t2_loader, t3_loader, scalers, token_dicts
