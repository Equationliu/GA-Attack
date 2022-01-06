import pandas as pd
import numpy as np
from PIL import Image 
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os

class MyCustomDataset(Dataset):
    def __init__(self, img_path = "images"):
        # Preprocess
        self.to_tensor = transforms.ToTensor()
        self.image_name = np.asarray(os.listdir(img_path))
        self.data_len = self.image_name.shape[0]
        self.img_path = img_path

    def __getitem__(self, index):
        single_image_name = self.image_name[index]
        img_as_img = Image.open(os.path.join(self.img_path, single_image_name)) 

        img_as_tensor = self.to_tensor(img_as_img)
        
        single_image_label = int(single_image_name.split('.')[0])

        return (img_as_tensor, single_image_label, single_image_name)

    def __len__(self):
        return self.data_len