import os
import numpy as np
import torch
from MovingMNIST import MovingMNIST

# npy_loaded = np.load('mnist_test_seq.npy')

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
train_set = MovingMNIST(root='.data/mnist', train=True, download=True)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=100,
                 shuffle=True)

print('==>>> total training batch number: {}'.format(len(train_loader)))

for seq, seq_target in train_loader:
    print('--- Sample')
    print('Input:  ', seq.shape)
    print('Target: ', seq_target.shape)
    break