import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

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
