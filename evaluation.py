import os
import numpy as np
import torchvision
import fid
import inception_score
import torch
from torch.utils.data import DataLoader

def evaluate(gen, n_noise, num_class, name_c, custom, num_img=10000, batch=10, save_path='./mid_test/img', save_img=True):
    with torch.no_grad():
        gen.eval()

        npc = num_img//num_class//batch

        eval_data = iter(DataLoader(custom, batch_size=batch))
        fake_images, real_images = [], []
        for i in range(num_img//batch):
            class_idx = i//npc
            class_name = name_c[class_idx].split()[2]

            label = torch.Tensor([class_idx]).cuda().long().repeat(batch)

            with torch.no_grad():
                fake = gen(torch.randn(batch, n_noise).cuda(), label)

            fake_images.append(fake.cpu().numpy())
            real_images.append(next(eval_data)[0].cpu().numpy())

            # Save generated images.
            if save_img:
                isave = i % npc
                save_name = save_path + '/img_%s_%04d.png' % (class_name, isave)

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torchvision.utils.save_image(fake[0], save_name, normalize=True)

        fake_images = np.concatenate(fake_images)
        real_images = np.concatenate(real_images)

        # Calculate FID scores
        print('Calculate FID scores')
        fid_score = fid.calculate_fid(fake_images, real_images, batch_size=batch)

        # Calculate Inception scores
        print('Calculate Inception scores')
        is_score = inception_score.inception_score(fake_images, batch_size=batch, splits=10)

        gen.train()
        return fid_score, is_score
