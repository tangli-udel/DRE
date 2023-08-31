import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from infinite_loader import InfiniteDataLoader

def split_valid(dataset, valid_ratio=0.2, s=0):
    n_val = int(np.floor(valid_ratio * len(dataset)))
    n_train = len(dataset) - n_val
    torch.manual_seed(s)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    return train_ds, val_ds

def make_dataset(root, tst_env, batch_size, transform, num_workers=8, valid_ratio=0.2, seed=0):
    environments = [f.name for f in os.scandir(root) if f.is_dir()]
    environments = sorted(environments)

    datasets = []
    for i, environment in enumerate(environments):
        path = os.path.join(root, environment)
        env_dataset = torchvision.datasets.ImageFolder(path, transform=transform)
        datasets.append(env_dataset)
    
    ds_tra = []
    ds_val = []
    for i, ds in enumerate(datasets):
        if i == tst_env:
            tstloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            tra, val = split_valid(ds, valid_ratio=valid_ratio, s=seed)
            ds_tra.append(tra)
            ds_val.append(val)
    
    val_length = 0
    valloaders = []
    for i, val in enumerate(ds_val):
        val_length += len(val)
        valloaders.append(torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers))
    
    ifloaders = []
    for i, tra in enumerate(ds_tra):
        ifloaders.append(InfiniteDataLoader(tra, weights=None, batch_size=batch_size, num_workers=num_workers))
    
    iters = []
    for i, ifloader in enumerate(ifloaders):
        iters.append(iter(ifloader))
    
    return datasets, iters, valloaders, tstloader, val_length
    

