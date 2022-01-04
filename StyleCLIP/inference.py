import os
from argparse import Namespace
from mapper.scripts.inference import run
from misc import ensuredir
meta_data = {
  'afro': ['afro', False, False, True],
  'angry': ['angry', False, False, True],
  'Beyonce': ['beyonce', False, False, False],
  'bobcut': ['bobcut', False, False, True],
  'bowlcut': ['bowlcut', False, False, True],
  'curly hair': ['curly_hair', False, False, True],
  'Hilary Clinton': ['hilary_clinton', False, False, False],
  'Jhonny Depp': ['depp', False, False, False],
  'mohawk': ['mohawk', False, False, True],
  'purple hair': ['purple_hair', False, False, False],
  'surprised': ['surprised', False, False, True],
  'Taylor Swift': ['taylor_swift', False, False, False],
  'trump': ['trump', False, False, False],
  'Mark Zuckerberg': ['zuckerberg', False, False, False]}
edit_types = ['afro', 'angry', 'Beyonce', 'bobcut',
              'bowlcut', 'curly hair', 'Hilary Clinton',
              'Jhonny Depp', 'mohawk', 'purple hair',
              'surprised', 'Taylor Swift', 'trump',
              'Mark Zuckerberg']
edit_type = edit_types[9]
edit_id = meta_data[edit_type][0]
img_name='me'
dataset='ffhq'
latents=f'{img_name}_latents.pt'
save_dir=f'results/{dataset}/{img_name}'
latent_path = f'{save_dir}/{latents}'
save_dir=f'{save_dir}/edit_type/{edit_type}'
ensuredir(save_dir)
#latent_path = '/mnt/nfs/sandbox/kinjo/GAN2Shape/data/celeba/latents/000386.pt'
num_images = 1
args = {
    'edit_type': edit_type,
    "work_in_stylespace": False,
    "exp_dir": save_dir,
    "checkpoint_path": f"pretrained/{edit_id}.pt",
    "couple_outputs": True,
    "mapper_type": "LevelsMapper",
    "no_coarse_mapper": meta_data[edit_type][1],
    "no_medium_mapper": meta_data[edit_type][2],
    "no_fine_mapper": meta_data[edit_type][3],
    "stylegan_size": 1024,
    "test_batch_size": 1,
    "latents_test_path": latent_path,
    "test_workers": 1,
    "n_images": num_images
}
run(Namespace(**args))
