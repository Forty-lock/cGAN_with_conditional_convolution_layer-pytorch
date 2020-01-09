"""Derived from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py"""  # NOQA
import torch
from torch.nn import functional as F

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, model, batch_size=32, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    model.eval()

    # Get predictions
    d0 = imgs.shape[0]
    preds = np.zeros((d0, 1000))

    n_batches = d0 // batch_size
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        batch = torch.from_numpy(imgs[start:end]).float().cuda()
        with torch.no_grad():
            _, pred = F.softmax(model(batch))

        preds[start:end] = pred.cpu().numpy()

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k::splits]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

