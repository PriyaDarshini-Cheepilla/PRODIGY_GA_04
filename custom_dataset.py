from torch.utils.data import Dataset
import os
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dirA, root_dirB, transform=None, subset_size=None):
        self.root_dirA = root_dirA
        self.root_dirB = root_dirB
        self.transform = transform

        # Get all images in the directories
        self.images_A = sorted(os.listdir(root_dirA))
        self.images_B = sorted(os.listdir(root_dirB))

        # If subset_size is provided, use only a subset of the images
        if subset_size:
            self.images_A = self.images_A[:subset_size]
            self.images_B = self.images_B[:subset_size]

    def __len__(self):
        return len(self.images_A)

    def __getitem__(self, idx):
        img_name_A = os.path.join(self.root_dirA, self.images_A[idx])
        img_name_B = os.path.join(self.root_dirB, self.images_B[idx])

        # Open images
        image_A = Image.open(img_name_A).convert('RGB')
        image_B = Image.open(img_name_B).convert('RGB')

        # Apply transformation if any
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

        return image_A, image_B
