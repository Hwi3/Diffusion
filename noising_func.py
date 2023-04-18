import torch
import torch.nn as nn
import numpy as np
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def beta_scheduelr(timestep, start=0.001, end =0.02):
    return torch.linspace(start,end,timestep)


def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

T = 300
betas = beta_scheduelr(timestep = T)
alphas = 1. - betas
#alphas_cumpord = [alphas[0], alphas[1]*[0], alphas[2]*[1], ..., alphas[300]*[299]]
alphas_cumprod = torch.cumprod(alphas,dim=0)
# alpha에서 오른쪽 끝에꺼 때버리고 한칸씩 당긴다음에 왼쪽 끝에 1 붙여줌
alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1],(1,0),value =1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)



# num_images = 10
# stepsize = int(T/num_images)


# image = torch.tensor(cv2.imread("C:\\Users\\Seunghwi\\Documents\\Diffusion\\archive\\Humans\\abc.jpg"),dtype=torch.float32)

# for idx in range(0, T, stepsize):
#     t = torch.Tensor([idx]).type(torch.int64)
#     image, noise = forward_diffusion_sample(image, t)
#     cv2.imwrite(f"{idx}.jpg",np.array(image))

# print()