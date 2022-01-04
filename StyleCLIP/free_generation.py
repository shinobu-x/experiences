import dlib
from argparse import Namespace
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from models.psp import pSp
from utils.alignment import align_face
from global_directions.manipulate import Manipulator
from optimization.run_optimization import main
from misc import ensuredir

device = 'cuda:3'
experiment_types = ['edit', 'free_generation']
experiment_type = experiment_types[1]
img_name='me'
dataset='ffhq'
save_dir=f'results/{dataset}/{img_name}'
latents=f'{img_name}_latents.pt'
latent_path = f'{save_dir}/{latents}'
save_dir=f'{save_dir}/experiment_type/{experiment_type}'
description = 'Joker muscular old male face pale white skin fangs slicked back hair' #@param {type:"string"}
lr=1e-3
optimization_steps = 10000
l2_lambda = 0.008
id_lambda = 0.005
stylespace = True
create_video = True
use_seed = False
seed =  0
ckpt='pretrained/stylegan2-ffhq-config-f.pt'
model='pretrained/model_ir_se50.pth'
results_dir=f'{save_dir}/{experiment_type}'
#ckpt='/mnt/nfs/sandbox/kinjo/GAN2Shape/checkpoints/stylegan2/ffhq.pt'
temp=f'{save_dir}/temp'
ensuredir(temp)
args = {
    "description": description,
    "ckpt": ckpt,
    "stylegan_size": 1024,
    "lr_rampup": 0.05,
    "lr": lr,
    "step": optimization_steps,
    "mode": experiment_type,
    "l2_lambda": l2_lambda,
    "id_lambda": id_lambda,
    'work_in_stylespace': stylespace,
    "latent_path": latent_path,
    "truncation": 0.7,
    "save_intermediate_image_every": 500 if create_video else 1,
    "results_dir": temp,
    "ir_se50_weights": model}
result = main(Namespace(**args))
result_image = to_pil_image(make_grid(result.detach().cpu(), normalize=True, scale_each=True, range=(-1, 1), padding=0))
result_image.save(f'{save_dir}/result.{ext}',ext.upper())
h,w=result_image.size
result_image.resize((h,w))
