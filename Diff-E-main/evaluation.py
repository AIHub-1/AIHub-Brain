from models import *
from utils import *

import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ema_pytorch import EMA
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

# set the device to use for evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# create an argument parser for the data loader path
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path to the data loader file')
parser.add_argument('--data_loader_path', type=str, help='path to the data loader file')

n_T = 1000
ddpm_dim = 128
encoder_dim = 256
fc_dim = 512
# Define model
num_classes = 13
channels = 64

encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
decoder = Decoder(
    in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim
).to(device)
fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
diffe = DiffE(encoder, decoder, fc).to(device)

# load the data loader from the file
args = parser.parse_args()
# load the pre-trained model from the file
diffe.load_state_dict(torch.load(args.model_path))

# load the data loader from the file
with open(args.data_loader_path, 'rb') as f:
    data_loader = pickle.load(f)

diffe.eval()
with torch.no_grad():
    labels = np.arange(0, 13)
    Y = []
    Y_hat = []
    for x, y in data_loader:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        encoder_out = diffe.encoder(x)
        y_hat = diffe.fc(encoder_out[1])
        y_hat = F.softmax(y_hat, dim=1)

        Y.append(y.detach().cpu())
        Y_hat.append(y_hat.detach().cpu())

    # List of tensors to tensor to numpy
    Y = torch.cat(Y, dim=0).numpy()  # (N, )
    Y_hat = torch.cat(Y_hat, dim=0).numpy()  # (N, 13): has to sum to 1 for each row

    # Accuracy and Confusion Matrix
    accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)

    print(f'Test accuracy: {accuracy:.2f}%')