import clip
import pickle
import torch
import lpips
from torchvision.transforms.functional import to_pil_image

pretrained='pretrained/stylegan3-t-ffhq-1024x1024.pkl'
with open(pretrained,'rb') as f:
    pretraiend = pickle.load(f)
G = pretraiend['G_ema'].to('cuda:1')
z = torch.rand(1,G.z_dim).to('cuda:1')
truncation = None
tensor_z = G(z, truncation)
tensor_z = tensor_z.to('cuda:1')
image_z = to_pil_image(tensor_z.squeeze(0))
w = G.mapping(z, truncation, truncation_psi=0.5, truncation_cutoff=8)
tensor_w = G.synthesis(w, noise_mode='const', force_fp32=True)
tensor_w = tensor_w.to('cuda:1')
image_w = to_pil_image(tensor_w.squeeze(0))
lpips_model = lpips.LPIPS('vgg')
clip_model, preprocess = clip.load('ViT-B/32')
preprocessed_z = preprocess(image_z).to('cuda:1')
preprocessed_w = preprocess(image_w).to('cuda:1')
lpips_model = lpips_model.to('cuda:1')
clip_model = clip_model.to('cuda:1')
lpips_loss = lpips_model(tensor_z, tensor_w)
features_z = clip_model.encode_image(preprocessed_z.unsqueeze(0)).float()
features_w = clip_model.encode_image(preprocessed_w.unsqueeze(0)).float()
features_z /= features_z.norm(dim=-1,keepdim=True)
features_w /= features_w.norm(dim=-1,keepdim=True)
with torch.inference_mode():
    similarity = features_z.cpu().numpy()@features_w.cpu().numpy().T
print(lpips_loss.item())
print(similarity[0][0])
