import numpy as np
import mat73
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Define a function to perform z-score normalization on the data
def zscore_norm(data):
    # Calculate the mean and standard deviation for each channel in each batch
    mean = torch.mean(data, dim=(1, 2))
    std = torch.std(data, dim=(1, 2))

    # Subtract the mean from each channel in each batch and divide by the standard deviation
    norm_data = (data - mean[:, None, None]) / std[:, None, None]

    return norm_data


# Define a function to perform min-max normalization on the data
def minmax_norm(data):
    # Calculate the minimum and maximum values for each channel and sequence in the batch
    min_vals = torch.min(data, dim=-1)[0]
    max_vals = torch.max(data, dim=-1)[0]

    # Scale the data to the range [0, 1]
    norm_data = (data - min_vals.unsqueeze(-1)) / (
        max_vals.unsqueeze(-1) - min_vals.unsqueeze(-1)
    )

    return norm_data


class EEGDataset(Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, X, Y, transform=None):
        "Initialization"
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.X)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Load data and get label
        x = self.X[index]
        y = self.Y[index]
        if self.transform:
            x = self.transform(x)
        return x.squeeze(), y


def load_data(root_dir, subject, session):
    filename = "s" + str(subject) + "_sess" + str(session)
    file = mat73.loadmat(
        root_dir + filename + ".mat"
    )  # Dictionary with keys Xi_train, Yi_train, Xi_test, Yi_test

    X = np.float32(file["X"])
    X = np.transpose(X, (2, 1, 0))
    Y = np.int_(file["y"]) - 1
    X, Y = zscore_norm(torch.from_numpy(X)), torch.from_numpy(Y)

    return X, Y


def get_dataloader(X, Y, batch_size, batch_size2, seed, shuffle=True):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=shuffle, stratify=Y, random_state=seed
    )

    training_set = EEGDataset(X_train, Y_train)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=shuffle)

    test_set = EEGDataset(X_test, Y_test)
    test_loader = DataLoader(test_set, batch_size=batch_size2, shuffle=False)

    return training_loader, test_loader
