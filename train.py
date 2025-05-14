import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.generator import Generator
from models.discriminator import Discriminator
from custom_dataset import CustomImageDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
trainA_path = 'data/train/trainA'
trainB_path = 'data/train/trainB'
subset_size = 20
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = CustomImageDataset(trainA_path, trainB_path, transform, subset_size=subset_size)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
generator = Generator().to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
def train(num_epochs=100):
    for epoch in range(num_epochs):
        for i, (real_A, real_B) in enumerate(train_loader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            optimizer_D.zero_grad()
            fake_B = generator(real_A)
            real_pred = discriminator(real_A, real_B)
            fake_pred = discriminator(real_A, fake_B.detach())
            real_labels = torch.ones_like(real_pred).to(device)
            fake_labels = torch.zeros_like(fake_pred).to(device)
            loss_D_real = adversarial_loss(real_pred, real_labels)
            loss_D_fake = adversarial_loss(fake_pred, fake_labels)
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            fake_pred = discriminator(real_A, fake_B)
            loss_G_adv = adversarial_loss(fake_pred, real_labels)
            loss_G_l1 = l1_loss(fake_B, real_B)
            loss_G = loss_G_adv + 100 * loss_G_l1
            loss_G.backward()
            optimizer_G.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Step [{i}/{len(train_loader)}] "
                      f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/discriminator_epoch_{epoch+1}.pth')
if __name__ == "__main__":
    train(num_epochs=5) 
