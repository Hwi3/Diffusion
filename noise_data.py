import data_loader
import noising_func
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import noising_func
import os
import cv2
import torch.nn.functional as F


os.environ['KMP_DUPLICATE_LIB_OK']='True'

BATCH_SIZE = 128
T = 300
data = data_loader.load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

image = next(iter(dataloader))[0]
#image = torch.tensor(cv2.imread("C:\\Users\\Seunghwi\\Documents\\Diffusion\\archive\\Humans\\1 (1).jpg"),dtype=torch.float32)
#image = image.view(3,image.shape[0],-1)



plt.figure(figsize=(15,15))
plt.axis('off')
num_images = 10
stepsize = int(T/num_images)

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, (idx//stepsize) + 1)
    image, noise = noising_func.forward_diffusion_sample(image, t)
    data_loader.show_tensor_image(image)
plt.show()