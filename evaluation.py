import os
import numpy as np
import torchvision
import fid
import inception_score
import torch
from inception_model import inception_v3

def evaluate(gen, n_noise, num_class, name_c, eval_data, time=3, save_path='./mid_test/img'):
    gen.eval()
    inception_model = inception_v3(pretrained=True).cuda()

    fid_score = np.zeros([time])
    is_score = np.zeros([time])
    for tt in range(time):
        fake_list, real_list = [], []
        for i in range(20000):
            class_idx = i//num_class
            class_name = name_c[class_idx].split()[2]

            z = torch.randn(1, n_noise).cuda()
            label = torch.Tensor([class_idx]).cuda().long()

            with torch.no_grad():
                fake = gen(z, label)

            fake_list.append((fake.cpu().numpy() + 1.0) / 2.0)
            real_list.append((next(eval_data)[0].cpu().numpy() + 1.0) / 2.0)

            # Save generated images.

            isave = i % num_class
            save_name = save_path + '/img_%04d_%s.png' % (isave, class_name)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torchvision.utils.save_image(fake[0], save_name, normalize=True)
        # Calculate FID scores

        fake_images = np.concatenate(fake_list)
        real_images = np.concatenate(real_list)

        mu_fake, sigma_fake = fid.calculate_activation_statistics(fake_images, inception_model, 20)
        mu_real, sigma_real = fid.calculate_activation_statistics(real_images, inception_model, 20)

        fid_score[tt] = fid.calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
        # Calculate Inception scores

        is_score[tt], _ = inception_score.inception_score(fake_images, inception_model, 20, resize=True, splits=10)
    gen.train()
    return fid_score, is_score
