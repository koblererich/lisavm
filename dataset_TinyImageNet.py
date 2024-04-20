import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os, glob
from torchvision.io import read_image, ImageReadMode


dataset_root = "./data/tiny-imagenet-200"


def get_id_dict():
    id_dict = {}
    for i, line in enumerate(open(os.path.join(dataset_root, 'wnids.txt'), 'r')):
      id_dict[line.replace('\n', '')] = i
    return id_dict


def get_cls_dict(id_dict):
    cls_dic = {}
    for line in open(os.path.join(dataset_root, "val/val_annotations.txt"), 'r'):
        a = line.split('\t')
        img, cls_id = a[0],a[1]
        cls_dic[img] = id_dict[cls_id]
    return cls_dic


class TrainTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        filenames = glob.glob(os.path.join(dataset_root, "train/*/*/*.JPEG"))
        filenames = [os.path.normpath(f) for f in filenames]
        self.images = [read_image(f, ImageReadMode.RGB).float()/255 for f in filenames]
        id_dict = get_id_dict()
        self.labels = [id_dict[f.split(os.sep)[3]] for f in filenames]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label


class TestTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        filenames = glob.glob(os.path.join(dataset_root, "val/images/*.JPEG"))
        filenames = [os.path.normpath(f) for f in filenames]
        self.images = [read_image(f, ImageReadMode.RGB).float()/255 for f in filenames]
        id_dict = get_id_dict()
        cls_dic = get_cls_dict(id_dict)
        self.labels = [cls_dic[f.split(os.sep)[-1]] for f in filenames]

        self.transform = transform
 
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image.type(torch.FloatTensor))
        return image, label
