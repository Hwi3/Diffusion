from torch.optim import Adam
from model import SimpleUnet
import torch
import noising_func
import torch.nn.functional as F
import data_loader
from torch.utils.data import DataLoader


model = SimpleUnet()


def get_loss(model, x_0, t):
    x_noisy, noise = noising_func.forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100 # Try more!


BATCH_SIZE = 128
T = 300
data = data_loader.load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()
      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      loss = get_loss(model, batch, t)
      loss.backward()
      optimizer.step()

      if step % 1 == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

torch.save(model.state_dict(), 'model_weights.pth')