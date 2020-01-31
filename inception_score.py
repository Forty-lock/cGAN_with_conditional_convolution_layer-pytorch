import torch
from torch.nn import functional as F

from torchvision.models.inception import inception_v3

import numpy as np


def cal_kl(p, q):
    return np.sum(p * np.log(p / q), axis=1)

def inception_score(imgs, batch_size=32, resize=True, splits=10):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = imgs.shape[0]

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model.eval()
    def get_pred(x):
        if resize:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    for i in range(N//batch_size):
        start = i * batch_size
        end = start + batch_size

        batch = torch.from_numpy(imgs[start:end]).float().cuda()

        preds[start:end] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k::splits]
        py = np.mean(part, axis=0, keepdims=True)
        scores = cal_kl(part, py)
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

