## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import sys
sys.path.append("/data/wangliyan/code/mycode/UHDDIP/")

import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
import utils
from natsort import natsorted
from glob import glob
from basicsr.models.archs.UHDDIP_arch import UHDDIP
from skimage import img_as_ubyte
import time

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

parser.add_argument('--input_dir', default='/data/wangliyan/dataset/UHD/', type=str, help='Directory of validation images')
parser.add_argument('--input_normal', default='/data/wangliyan/dataset/UHD/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results_UHDLL/', type=str, help='Directory for results')
parser.add_argument('--weights', default='/data/wangliyan/code/mycode/UHD/C2F-DFT-main/experiments/UHDIR_UHD-LL_6w/models/net_g_latest.pth', type=str, help='Path to weights')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/train_UHDDIP.yml'

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################
model_restoration = UHDDIP(**x['network_g'])
checkpoint = torch.load(args.weights)

model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

datasets = ['testing_set']
test_times = []
for dataset in datasets:
    result_dir  = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    inp_dir = os.path.join(args.input_dir, dataset, 'input')
    inp_normal = os.path.join(args.input_normal, dataset, 'cond/normal')

    files = natsorted(glob(os.path.join(inp_dir, '*.JPG')) + glob(os.path.join(inp_dir, '*.png')))
    files_n = natsorted(glob(os.path.join(inp_normal, '*.png')) + glob(os.path.join(inp_normal, '*.jpg')))
    with torch.no_grad():
        for file_, normal in tqdm(zip(files, files_n)):
        # for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_))/255.
            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()

            normal = np.float32(utils.load_img(normal)) / 255.
            normal = torch.from_numpy(normal).permute(2, 0, 1)
            normal = normal.unsqueeze(0).cuda()

            tic = time.time()
            restored, _ = model_restoration(input_, normal)

            toc = time.time()
            test_times.append(toc - tic)

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.JPG')), img_as_ubyte(restored))

    print(f"average test time: {np.mean(test_times):.4f}")