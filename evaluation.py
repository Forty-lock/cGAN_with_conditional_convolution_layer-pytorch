import os
import numpy as np
import torchvision
import fid
import inception_score
import torch
from torch.utils.data import DataLoader
from inception_model import inception_v3

def evaluate(gen, n_noise, num_class, name_c, custom, time=3, num_img=10000, batch=1, save_path='./mid_test/img', save_img=True):
    with torch.no_grad():
        gen.eval()
        inception_model = inception_v3(pretrained=True, transform_input=True).cuda()

        npc = num_img//num_class//batch

        fid_score = []
        is_score = []
        for tt in range(time):
            eval_data = iter(DataLoader(custom, batch_size=batch))
            fake_images, real_images = [], []
            for i in range(num_img//batch):
                class_idx = i//npc
                class_name = name_c[class_idx].split()[2]

                z = torch.randn(batch, n_noise).cuda()
                label = torch.Tensor([class_idx]).cuda().long().repeat(batch)

                with torch.no_grad():
                    fake = gen(z, label)

                fake_images.append((fake.cpu().numpy() + 1.0) / 2.0)
                real_images.append((next(eval_data)[0].cpu().numpy() + 1.0) / 2.0)

                # Save generated images.
                if save_img:
                    if tt == 0:
                        isave = i % npc
                        save_name = save_path + '/img_%04d_%s.png' % (isave, class_name)

                        if not os.path.isdir(save_path):
                            os.makedirs(save_path)
                        torchvision.utils.save_image(fake[0], save_name, normalize=True)

            fake_images = np.concatenate(fake_images)
            real_images = np.concatenate(real_images)

            # Calculate FID scores
            mu_fake, sigma_fake = fid.calculate_activation_statistics(fake_images, inception_model, 50)
            mu_real, sigma_real = fid.calculate_activation_statistics(real_images, inception_model, 50)

            fid_score.append(fid.calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real))

            # Calculate Inception scores
            is_score.append(inception_score.inception_score(fake_images, inception_model, 50, splits=10))

        gen.train()
        return fid_score, is_score
