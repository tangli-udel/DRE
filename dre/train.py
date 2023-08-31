from pdb import Restart
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from tqdm import tqdm
import numpy as np

sys.path.append('./dre/')
from datasets import make_dataset
from mixuploss import MixupLoss
from networks import ResNet

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser(description='Distributionally Robust Explanations.')
parser.add_argument('--dataset', default='terra_incognita', type=str, help='terra_incognita, vlcs')
parser.add_argument('--model', default='DRE', type=str, help='ERM, DRE')
parser.add_argument('--tst_env', default=0, type=int, help='testing distribution (default: 0)')
parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default: 16)')
parser.add_argument('--seed', default=0, type=int, help='random seed (default: 0)')
parser.add_argument('--gpu_id', default="0", type=str, help='gpu id')
args = parser.parse_args()

# Choose GPU Devices
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(args.seed)

batch_size = args.batch_size
steps = 5000
if args.model == 'DRE':
    warmup_steps = 0
elif args.model == "ERM":
    warmup_steps = 5001
lr = 5e-5
check_freq = 200
exp_freq = 1
if args.dataset == 'terra_incognita':
    root = '../data/terra_incognita/'
elif args.dataset == 'vlcs':
    root = '../data/VLCS/'
tst_env = args.tst_env

exp_weight = 1.0
sparse_weight = 0.5
mix_ce_weight = 0.1

def validate_mixup(model, device, val_loader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            ce_loss = F.cross_entropy(output, target) 
            exp_loss, sparse_loss, mix_ce_loss = MixupLoss(inputs, targets, net, num_classes=len(datasets[0].classes), device=device)
            val_loss += ce_loss + exp_weight*exp_loss + sparse_weight*sparse_loss + mix_ce_weight*mix_ce_loss
        val_loss /= len(val_loader.dataset)
    return val_loss.item()

def validate(model, device, val_loader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            ce_loss = F.cross_entropy(output, target) 
            val_loss += ce_loss
        val_loss /= len(val_loader.dataset)
    return val_loss.item()


transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

datasets, iters, valloaders, tstloader, val_length = make_dataset(root, tst_env=tst_env, batch_size=batch_size, transform=augment_transform, seed=0)

net = ResNet()
net= nn.DataParallel(net)
net.module.network.fc = nn.Linear(net.module.network.fc.in_features, 10)
net = net.to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5, threshold=0.1, threshold_mode='rel')


best_loss = 1.0e10
best_acc = 0.0
step_loss = 0.0
step_ce_loss = 0.0
step_exp_loss = 0.0
step_sparse_loss = 0.0
step_mix_ce_loss = 0.0

for step in tqdm(range(steps), ncols=100):
    inputs = []
    targets = []
    for i, it in enumerate(iters):
        input, target = next(it)
        inputs.append(input)
        targets.append(target)
    inputs = torch.cat(inputs, dim=0).to(device)
    targets = torch.cat(targets, dim=0).to(device)

    net.train()
    optimizer.zero_grad()
    outputs = net(inputs)
    ce_loss = F.cross_entropy(outputs, targets)
    if (step >= warmup_steps) & (step % exp_freq == 0):
        exp_loss, sparse_loss, mix_ce_loss = MixupLoss(inputs, targets, net, num_classes=len(datasets[0].classes), device=device)
        loss = ce_loss + exp_weight*exp_loss + sparse_weight*sparse_loss + mix_ce_weight*mix_ce_loss
        loss.backward()

        step_loss += loss.item()
        step_ce_loss += ce_loss.item()
        step_exp_loss += exp_weight*exp_loss.item()
        step_sparse_loss += sparse_weight*sparse_loss.item()
        step_mix_ce_loss += mix_ce_weight*mix_ce_loss.item()
    else:
        ce_loss.backward()
        step_loss += ce_loss.item()
        
    optimizer.step()


    if step % check_freq == 0:
        val_loss = 0.0
        for i, val in enumerate(valloaders):
            val_loss += (len(val.dataset)/val_length) * validate(net, device, val)

        step_loss = step_loss/check_freq
        step_ce_loss = step_ce_loss/check_freq
        step_exp_loss = step_exp_loss/(check_freq/exp_freq)
        step_sparse_loss = step_sparse_loss/(check_freq/exp_freq)
        step_mix_ce_loss = step_mix_ce_loss/(check_freq/exp_freq)

        print('\nStep [' + str(step) + '] Training Loss: ' + str(step_loss) 
                + '   CELoss: ' + str(step_ce_loss)
                + '   ExpLoss: ' + str(step_exp_loss)
                + '   SparseLoss: ' + str(step_sparse_loss)
                + '   MixCELoss: ' + str(step_mix_ce_loss))
        print('Step [' + str(step) + '] Validation Loss:' + str(val_loss))
            
        with open("logs/lr.log","a") as f:
            f.write(str(optimizer.state_dict()['param_groups'][0]['lr']) + '\n')
        
        with open("logs/trn_loss.log","a") as f:
            f.write(str(step) + ' ' + str(step_loss) + '\n')
            
        with open("logs/val_loss.log","a") as f:
            f.write(str(step) + ' ' + str(val_loss) + '\n')
        
        step_loss = 0.0
        step_ce_loss = 0.0
        step_exp_loss = 0.0
        step_sparse_loss = 0.0
        step_mix_ce_loss = 0.0

        scheduler.step(val_loss)


        torch.save(net.state_dict(), './ckpts/step_{}.pth'.format(step))
        if val_loss < best_loss:
            torch.save(net.state_dict(), './ckpts/best_val_model.pth')
            print('val Model saved.')
            best_loss = val_loss