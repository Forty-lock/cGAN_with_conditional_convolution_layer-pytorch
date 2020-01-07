from torch.utils.data import Dataset
import torchvision.transforms as trans
import pickle
import torch

class CustomDataset(Dataset):
    def __init__(self, dataset_root):
        with open(dataset_root + 'tiny_ImageNet128.pkl', 'rb') as pickle_file:
            self.data_list = pickle.load(pickle_file)
        with open(dataset_root + 'tiny_ImageNet128_label.pkl', 'rb') as pickle_file:
            self.label_list = pickle.load(pickle_file)
        self.img_transform1 = trans.Compose([trans.ToTensor(),
                                            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ])
        self.img_transform2 = trans.Compose([trans.ToTensor(),
                                             trans.Lambda(lambda x: x.repeat(3, 1, 1)),
                                             trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             ])

    def __getitem__(self, idx):
        img = self.data_list[idx]
        gt = torch.Tensor([int(self.label_list[idx]) - 1])

        if img.ndim == 2:
            img = self.img_transform2(img)
        else:
            img = self.img_transform1(img)

        return img.cuda(), gt[0].cuda().long()

    def __len__(self):
        return len(self.data_list)

