
import os
from PIL import Image
from os.path import join
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class SimpleDataset(Dataset):

    def __init__(self, folder_data, transforms=None, split = None):

        super(SimpleDataset, self).__init__()
        self.folder_data = folder_data
        self.transforms = transforms
        self.list_img = os.listdir(self.folder_data)
        self.list_img_all = self.list_img
        self.split = split
        if self.split is not None:
            self.train_list, self.valid_list = train_test_split(self.list_img, test_size = 0.2, random_state = 42)
            if split == 'valid':
                self.list_img = self.valid_list
            elif split == 'train':
                self.list_img = self.train_list
            else:
                raise Exception("this split is not supported")

    def __getitem__(self, idx):
        filename = self.list_img[idx]
        img = Image.open(join(self.folder_data, filename))

        if self.transforms is not None:
            img = self.transforms(img)

        return img, filename

    def __len__(self):
        return len(self.list_img)