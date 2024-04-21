"""
General test script for HiSD. 
"""

# if type is 'latent-guided', you need to specifc 'attribute' and 'seed' (None means random).
# Otherwise if type is 'reference-guided', you need to specifc the 'reference' file path.
steps = [
    #{'type': 'latent-guided', 'tag': 0, 'attribute': 0, 'seed': None},
    {'type': 'latent-guided', 'tag': 0, 'attribute': 1, 'seed': None},
    {'type': 'reference-guided', 'tag': 0, 'reference': '/data1/huangwenmin/CelebAMask-HQ/val/bangs_with/1118.jpg'},
]

from utils import get_config
from trainer import SDGAN_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import time
os.environ["CUDA_VISIBLE_DEVICES"]="7"
parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='/home/huangwenmin/SDGAN/configs/celeba-hq_256.yaml', help='Path to the config file.')
parser.add_argument('--checkpoint', type=str, default='/home/huangwenmin/SDGAN/core/outputs/celeba-hq_256/checkpoints/gen.pt')
parser.add_argument('--input_path', type=str, default='/data1/huangwenmin/CelebAMask-HQ/val/bangs_without')
parser.add_argument('--output_path', type=str, default='./test')
parser.add_argument('--seed', type=int, default=None)

opts = parser.parse_args()

os.makedirs(opts.output_path, exist_ok=True)

config = get_config(opts.config)
noise_dim = config['noise_dim']
trainer = SDGAN_Trainer(config)
state_dict = torch.load(opts.checkpoint)
trainer.models.gen.load_state_dict(state_dict['gen_test'])
trainer.models.gen.cuda()

E = trainer.models.gen.encode
T = trainer.models.gen.translate
G = trainer.models.gen.decode
M = trainer.models.gen.map
F = trainer.models.gen.extract

filename = time.time()
transform = transforms.Compose([transforms.Resize(config['new_size']),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if os.path.isfile(opts.input_path):
    inputs = [opts.input_path]
else:
    inputs = [os.path.join(opts.input_path, file_name) for file_name in os.listdir(opts.input_path)]

with torch.no_grad():
    for input in inputs:
        x = transform(Image.open(input).convert('RGB')).unsqueeze(0).cuda()
        c = E(x)
        for j in range(len(steps)):
            step = steps[j]
            s = F(x, step['tag'])
            if step['type'] == 'latent-guided':
                if step['seed'] is not None:
                    torch.manual_seed(step['seed'])
                    torch.cuda.manual_seed(step['seed']) 

                z = torch.randn(1, noise_dim).cuda()
                s_trg = M(z, step['tag'], step['attribute'])
                c_trg = T(c, s, s_trg, step['tag'])

            elif step['type'] == 'reference-guided':
                reference = transform(Image.open(step['reference']).convert('RGB')).unsqueeze(0).cuda()
                s_trg = F(reference, step['tag'])
                c_trg = T(c, s, s_trg[:, step['tag']], step['tag'])
            

            
            x_trg = G(c_trg)
            vutils.save_image(((x_trg + 1)/ 2).data, os.path.join(opts.output_path, f'{os.path.splitext(os.path.basename(input))[0]}_{str(j)}_output.jpg'), padding=0)

