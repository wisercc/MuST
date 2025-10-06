from __future__ import print_function

import pickle

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

import os

class BCI2aDataset(Dataset):
    def __init__(self, mat_file_path):
        """
        Args:
            mat_file_path (str): Path to the .mat file containing the data and labels.
        """
        # Load .mat file
        mat = sio.loadmat(mat_file_path)

        # Extract data and labels from .mat file
        self.data = mat['data']
        self.labels = mat['label'] - 1
        self.labels = self.labels.squeeze()
        self.data = np.transpose(self.data, [2, 1, 0])

        self.data = self.normalize_channels(self.data)
        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def normalize_channels(self, data):
        """
        Normalize the data channel-wise.
        Args:
            data (numpy array): EEG data
        Returns:
            normalized_data (numpy array): Normalized EEG data
        """
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        normalized_data = (data - mean) / (std + 0.00001)
        return normalized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        return data, label

class SeedDataset(Dataset):
    def __init__(self, sub, type, file_path=r'/disk2/zhaokui/data/seed_1s/'):
        """
        Args:
            sub (int): Subject identifier, Integer
            type (str): Type of the experiment, String
            file_path (str): Path to the directory containing the .npy files
        """
        all_data = []
        all_labels = []

        # Iterate through all three sessions
        for session in range(1, 4):  # Assuming sessions are 1, 2, and 3
            # Load .npy files
            base_path = file_path + 'S' + str(sub) + '_session' + str(session) + type
            data_path = base_path + '.npy'
            label_path = base_path + '_label.npy'
            data = np.load(data_path)
            data = np.transpose(data, [2, 1, 0])
            label = np.load(label_path) + 1

            all_data.append(data)
            all_labels.append(label)

        # Concatenate all sessions' data and labels
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Convert to PyTorch tensors
        self.data = torch.tensor(all_data, dtype=torch.float32)
        self.labels = torch.tensor(all_labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        return data, label

class SleepDataset(Dataset):
    def __init__(self, file_path=r'/disk2/zhaokui/data/SleepEDF/sleepEDF20/'):
        """
        Args:
            file_path (str): Path to the directory containing the .npz files
        """
        all_files = os.listdir(file_path)
        npz_files = [file for file in all_files if file.endswith('.npz')]

        data = []
        label = []
        for f in npz_files:
            npz = np.load(file_path + f, allow_pickle=True)
            data.append(npz['x'])
            label.append(npz['y'])

        data = np.concatenate(data, axis=0)
        self.data = self.normalize_channels(np.transpose(data, [0, 2, 1]))


        self.labels = np.concatenate(label, axis=0)

        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def normalize_channels(self, data):
        """
        Normalize the data channel-wise.
        Args:
            data (numpy array): EEG data
        Returns:
            normalized_data (numpy array): Normalized EEG data
        """
        mean = np.mean(data, keepdims=True)
        std = np.std(data, keepdims=True)
        normalized_data = (data - mean) / (std + 0.00001)
        return normalized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        return data, label


class SeizureDataset(Dataset):
    def __init__(self, file_path='/disk2/zhaokui/data/bonn/'):
        """
        Args:
            file_path (str): Path to the directory containing the .mat files
        """
        norm_sub = ['A', 'B']
        abnorm_sub = ['C', 'D', 'E']

        norm_data = []
        abnorm_data = []
        for i in norm_sub:
            path = file_path + i + '.mat'
            data = sio.loadmat(path)
            norm_data.append(data['data'])

        for i in abnorm_sub:
            path = file_path + i + '.mat'
            data = sio.loadmat(path)
            abnorm_data.append(data['data'])

        data = norm_data + abnorm_data

        label = np.concatenate((np.zeros(200), np.ones(300)))
        data = np.concatenate(data, axis=2)

        data = np.transpose(data, [2, 1, 0])

        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)

        data = data[indices]
        label = label[indices]

        data = self.normalize_channels(data)

        self.data = torch.tensor(data[..., 0:4096], dtype=torch.float32)
        self.labels = torch.tensor(label, dtype=torch.long)


    def normalize_channels(self, data):
        """
        Normalize the data channel-wise.
        Args:
            data (numpy array): EEG data
        Returns:
            normalized_data (numpy array): Normalized EEG data
        """
        mean = np.mean(data, keepdims=True)
        std = np.std(data, keepdims=True)
        normalized_data = (data - mean) / (std + 0.00001)
        return normalized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        return data, label


class FatigDataset(Dataset):
    def __init__(self, file_path='/disk2/zhaokui/data/fatig/data_eeg_FATIG_FTG/', sub=0):
        """
        Args:
            file_path (str): Path to the directory containing the .pkl files
            sub (int): Subject identifier, Integer
        """
        path = file_path + 'sub' + str(sub) + '.pkl'

        with open(path, 'rb') as file:
            pkl = pickle.load(file)
        data = np.concatenate(pkl['data'], axis=0)
        label = np.concatenate(pkl['label'], axis=0)

        data = self.normalize_channels(data)
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(label, dtype=torch.long)

    def normalize_channels(self, data):
        """
        Normalize the data channel-wise.
        Args:
            data (numpy array): EEG data
        Returns:
            normalized_data (numpy array): Normalized EEG data
        """
        mean = np.mean(data, keepdims=True)
        std = np.std(data, keepdims=True)
        normalized_data = (data - mean) / (std + 0.00001)
        return normalized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        return data, label
