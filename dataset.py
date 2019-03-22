import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from PIL import Image, ImageFile
from torchvision.transforms import transforms

transformation = transforms.Compose([
    # transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SegDataset(Dataset):
    def __init__(self, dir_path):
        self.image_dir = os.path.join(dir_path, 'images')
        self.label_dir = os.path.join(dir_path, 'labels')
        self.all_names = [ f for f in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, idx):
        name = self.all_names[idx]
        image = Image.open(os.path.join(self.image_dir, name))
        label = Image.open(os.path.join(self.label_dir, name))

        sample = [transformation(image), transformation(label)]
        return sample