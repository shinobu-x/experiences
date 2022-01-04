from optimization.run_optimization import main
from argparse import Namespace
from misc import ensuredir

experiment_types = ['edit', 'free_generation']
experiment_type = experiment_types[0]
img_name='me'
dataset='ffhq'
latents=f'{img_name}_latents.pt'
save_dir=f'results/{dataset}/{img_name}'
latent_path = f'{save_dir}/{latents}'
save_dir=f'{save_dir}/experiment_type/{experiment_type}'
ensuredir(save_dir)
description = 'A person with purple hair'
optimization_steps = 40
l2_lambda = 0.008
id_lambda = 0.005
stylespace = False
create_video = True
use_seed = True
seed = 1
ckpt='pretrained/stylegan2-ffhq-config-f.pt'
weights= "pretrained/model_ir_se50.pth"
args = {
    "description": description,
    "ckpt": ckpt,
    "stylegan_size": 1024,
    "lr_rampup": 0.05,
    "lr": 0.1,
    "step": optimization_steps,
    "mode": experiment_type,
    "l2_lambda": l2_lambda,
    "id_lambda": id_lambda,
    'work_in_stylespace': stylespace,
    "latent_path": latent_path,
    "truncation": 0.7,
    "save_intermediate_image_every": 1 if create_video else 20,
    "results_dir": save_dir,
    "ir_se50_weights": weights
}
result = main(Namespace(**args))
