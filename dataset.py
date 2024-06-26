import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")
plt.ion()

import opendatasets as od

dataset_url = 'https://www.kaggle.com/datasets/vaibhavrmankar/colour-the-flower-gan-data/code'
dataset_dir = '/content/colour-the-flower-gan-data'
od.download(dataset_url, root=dataset_dir, extract=True)

# Set up directory paths
train_dir = os.path.abspath(os.path.join('.', 'colour-the-flower-gan-data', 'Data'))

if os.path.exists(dataset_dir) and os.path.exists(train_dir):
    print("The dataset and train directory exist.")
else:
    print("The dataset or train directory does not exist.")

os.listdir(train_dir)

image_path = os.path.join(train_dir, 'train/1.jpg')
Image.open(image_path)


class PairedFlowerDataset(Dataset):
    def __init__(self, root_dir, transform_color,transform_grayscale):
        self.root_dir = root_dir
        self.transform_color = transform_color
        self.transform_grayscale = transform_grayscale
        self.image_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')])

    def __len__(self):
        # We assume each image is paired, so we divide by 2
        return len(self.image_paths) // 2

    def __getitem__(self, idx):
    # Load the image (assuming it's a single image with color on the left and grayscale on the right)
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

    # Crop the color part from the left side of the image
        width, height = image.size
        color_image = image.crop((0, 0, width // 2, height))  # crop the left half of the image

    # Convert the right half of the image to grayscale
        grayscale_image = image.crop((width // 2, 0, width, height))  # crop the right half of the image
        grayscale_image = grayscale_image.convert('L')

    # Apply transformations to the color and grayscale images
        color_image = self.transform_color(color_image)
        grayscale_image = self.transform_grayscale(grayscale_image)

        return color_image, grayscale_image
    
transform_color = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_grayscale = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = PairedFlowerDataset(train_dir, transform_color, transform_grayscale)
dataloader = DataLoader(dataset, batch_size=30, shuffle=True)

color_image, _ = dataset[9]
color_image = color_image.numpy().transpose((1, 2, 0))
# Display only the color image
plt.imshow(color_image)
plt.title(f"Transformed Image")
plt.show()

_,grayscale_image = dataset[9]
grayscale_image = grayscale_image.numpy().squeeze()
plt.imshow(grayscale_image)
plt.title(f"Transformed Image")
plt.show()