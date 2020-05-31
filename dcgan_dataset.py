
import os
from PIL import Image
from os.path import join
from torch.utils.data import Dataset


class SimpleDataset(Dataset):

    def __init__(self, folder_data, transforms=None):

        super(SimpleDataset, self).__init__()
        self.folder_data = folder_data
        self.transforms = transforms
        self.list_img = os.listdir(self.folder_data)

    def __getitem__(self, idx):
        filename = self.list_img[idx]
        img = Image.open(join(self.folder_data, filename))

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.list_img)