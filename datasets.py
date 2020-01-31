from torch.utils.data import Dataset
import torchvision.transforms as trans
import numpy as np
import torch
import glob
import cv2
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, dataset_root):
        self.data_list = glob.glob(dataset_root + '/*/*.jpeg')
        self.label_name, self.num_label = read_label_list(dataset_root + '/labels_image_net.txt')
        self.img_transform = trans.Compose([trans.ToTensor(),
                                            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])

    def __getitem__(self, idx):
        # Loadimage = cv2.imread(self.data_list[idx])
        # Loadimage = cv2.cvtColor(Loadimage, cv2.COLOR_BGR2RGB)
        Loadimage = Image.open(self.data_list[idx])
        Loadimage = np.asarray(Loadimage)

        idx_class = self.data_list[idx].split('\\')[-2]
        label = torch.Tensor([int(idx_class) - 1])

        if Loadimage.ndim == 2:
            Loadimage = np.expand_dims(Loadimage, 2)
            Loadimage = np.tile(Loadimage, (1, 1, 3))

        img = self.img_transform(Loadimage)

        return img, label[0].long()

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
