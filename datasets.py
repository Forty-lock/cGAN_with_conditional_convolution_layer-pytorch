from torch.utils.data import Dataset
import torchvision.transforms as trans
import pickle
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, dataset_root):
        with open(dataset_root + 'tiny_ImageNet128.pkl', 'rb') as pickle_file:
            self.data_list = pickle.load(pickle_file)
        with open(dataset_root + 'tiny_ImageNet128_label.pkl', 'rb') as pickle_file:
            self.label_list = pickle.load(pickle_file)
        self.label_name, self.num_label = read_label_list(dataset_root + 'labels_tiny.txt')
        self.img_transform = trans.Compose([trans.ToTensor(),
                                            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])

    def __getitem__(self, idx):
        Loadimage = self.data_list[idx]
        gt = torch.Tensor([int(self.label_list[idx]) - 1])

        if Loadimage.ndim == 2:
            Loadimage = np.expand_dims(Loadimage, 2)
            Loadimage = np.tile(Loadimage, (1, 1, 3))

        img = self.img_transform(Loadimage)

        return img, gt[0].long()

    def __len__(self):
        return len(self.data_list)

def read_label_list(image_list_file):
    f = open(image_list_file, 'r')
    filenames1 = []
    Total_Image_Num = 0

    for line in f:
        filename1 = line[:-1]

        filenames1.append(filename1)

        Total_Image_Num = Total_Image_Num + 1

    return filenames1, Total_Image_Num
