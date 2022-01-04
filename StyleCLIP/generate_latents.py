import dlib
from argparse import Namespace
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from models.psp import pSp
from utils.alignment import align_face
from global_directions.manipulate import Manipulator
from dlib import shape_predictor
from misc import ensuredir

def run_alignment(image_path):
    landmark='landmarks/shape_predictor_68_face.dat'
    predictor = shape_predictor(landmark)
    aligned_image = align_face(filepath=image_path,predictor=predictor)
    return aligned_image

device='cuda:3'
img_name='me'
dataset='ffhq'
save_dir=f'results/{dataset}/{img_name}'
ensuredir(save_dir)
experiment_type='ffhq_enocde'
model_path= 'pretrained/e4e_ffhq_encode.pt'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts['device']=device
opts = Namespace(**opts)
model = pSp(opts)
model.eval()
model = model.to(device)
print('Model successfully loaded!')
ext='jpeg'
image_path=f'samples/{img_name}.{ext}'
img=Image.open(image_path)
img=img.convert('RGB')
input = run_alignment(image_path)
input.resize((256,256))
img.save(f'{save_dir}/{img_name}_original_input.{ext}',ext.upper())
input.save(f'{save_dir}/{img_name}_aligned_input.{ext}',ext.upper())
transformed_image=transform(input)
transformed_image=transformed_image.unsqueeze(0)
transformed_image=transformed_image.to(device).float()
with torch.no_grad():
    images, latents, style_vector = model(transformed_image, randomize_noise=False,return_latents=True)
    if experiment_type=='cars_encode':
        images = images[:,:,32:224,:]
image=images[0]
latent=latents[0]
img=to_pil_image(image)
img.save(f'{save_dir}/{img_name}_results.{ext}',ext.upper())
torch.save(latents,f'{save_dir}/{img_name}_latents.pt')
