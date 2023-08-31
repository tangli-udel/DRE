import numpy as np
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from utils import *
from evaluation import CausalMetric, auc, gkern
from pytorch_grad_cam import GradCAM
import argparse

import sys
sys.path.append('../../../dre/')
from networks import ResNet


parser = argparse.ArgumentParser(description='PyTorch AUC Metric Evaluation')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
parser.add_argument('--ckpt-path', dest='ckpt_path', type=str, help='path to checkpoint file')
parser.add_argument('--root', dest='root', type=str, help='path to dataset')


args = parser.parse_args()

cudnn.benchmark = True

scores = {'del': [], 'ins': []}

net = ResNet()
net= nn.DataParallel(net)
net.module.network.fc = nn.Linear(net.module.network.fc.in_features, 10)

# remove the module prefix if model was saved with DataParallel
# state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# load params
net.load_state_dict(torch.load(args.ckpt_path))

target_layer = net.module.network.layer4[-1]
cam = GradCAM(model=net.module, target_layer=target_layer, use_cuda=True)

batch_size = 100

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

data_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(args.root, transform=augment_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=True)

num_sample = int(len(data_loader.dataset) / 10)

def get_auc_per_data_subset(range_index, net, cam, num=num_sample):
    net = net.train()
    data_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(args.root, transform=augment_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=True, sampler=RangeSampler(range(num * range_index, num * (range_index + 1))))

    images = []
    targets = []
    gcam_exp = []
    for j, (img, trg) in enumerate(tqdm(data_loader, total=len(data_loader), desc='Loading images')):
        grayscale_gradcam = cam(input_tensor=img, target_category=trg)
        for k in range(batch_size):
            images.append(img[k])
            targets.append(trg[k])
            gcam_exp.append(grayscale_gradcam[k])

    images = torch.stack(images).cpu().numpy()
    gcam_exp = np.stack(gcam_exp)
    images = np.asarray(images)
    gcam_exp = np.asarray(gcam_exp)

    images = images.reshape((-1, 3, 224, 224))
    gcam_exp = gcam_exp.reshape((-1, 224, 224))

    model = nn.Sequential(net.module, nn.Softmax(dim=1))
    model = model.eval()
    model = model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    ddp_model = nn.DataParallel(model)

    # we use blur as the substrate function
    klen = 11
    ksig = 5
    kern = gkern(klen, ksig)
    # Function that blurs input image
    blur = lambda x: F.conv2d(x, kern, padding=klen // 2)

    insertion = CausalMetric(ddp_model, 'ins', 224 * 8, substrate_fn=blur)
    h = insertion.evaluate(torch.from_numpy(images.astype('float32')), gcam_exp, batch_size)

    model = model.train()
    for p in model.parameters():
        p.requires_grad = True
        
    return auc(h.mean(1))


# we process the dataset in 10 subsets due to limited memory
for i in range(10):
    auc_score = get_auc_per_data_subset(i, net, cam)
    scores['ins'].append(auc_score)
    print('Finished evaluating the insertion metrics...')

print('----------------------------------------------------------------')
print('Final:\nInsertion - {:.5f}'.format(np.mean(scores['ins'])))
