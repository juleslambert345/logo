
import os
from PIL import Image
from os.path import join
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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


class LabelDataset(Dataset):

    def __init__(self, folder_data, transforms=None, split = None, encoder = None):

        super(LabelDataset, self).__init__()
        self.folder_data = folder_data
        self.transforms = transforms
        self.list_class = os.listdir(self.folder_data)
        self.list_img = []
        self.list_class_of_image = []
        self.list_class_encoding = []

        for class_name in self.list_class:
            filenames = os.listdir(join(self.folder_data, class_name))
            self.list_img+=filenames
            self.list_class_of_image += [class_name]*len(filenames)

        list_of_list = [[class_name] for class_name in self.list_class_of_image]
        if encoder is None:
            self.encoder = OneHotEncoder(handle_unknown='ignore')
            self.encoding_matrix = self.encoder.fit_transform(list_of_list).toarray()
        else:
            self.encoder = encoder
            self.encoding_matrix = self.encoder.transform(list_of_list).toarray()

        self.number_class = len(set(self.list_class_of_image))


        self.list_img_all = self.list_img
        self.list_class_all = self.list_class_of_image
        self.encoding_matrix_all = self.encoding_matrix.copy()
        self.split = split
        if self.split is not None:
            self.train_list, self.valid_list, self.train_class, self.valid_class, self.train_encoding, self.valid_encoding = train_test_split(self.list_img, self.list_class_of_image , self.encoding_matrix,  test_size = 0.2, random_state = 42)
            if split == 'valid':
                self.list_img = self.valid_list
                self.list_class_of_image = self.valid_class
                self.encoding_matrix = self.valid_encoding
            elif split == 'train':
                self.list_img = self.train_list
                self.list_class_of_image = self.train_class
                self.encoding_matrix = self.train_encoding
            else:
                raise Exception("this split is not supported")

    def __getitem__(self, idx):
        filename = self.list_img[idx]
        cluster_name = self.list_class_of_image[idx]
        img = Image.open(join(self.folder_data, cluster_name, filename))

        if self.transforms is not None:
            img = self.transforms(img)

        class_vector = self.encoding_matrix[idx]
        class_name = self.list_class_of_image[idx]

        return img, filename, class_vector, class_name

    def __len__(self):
        return len(self.list_img)