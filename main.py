import torch
import torch.optim as optim
import time
import module as mm
import os
import Read_Image_List as Ril
from datasets import CustomDataset
from torch.utils.data import DataLoader
import cv2
import numpy as np

Height = 128
Width = 128
batch_size = 32
n_noise = 128

GD_ratio = 5

description = 'v1'

save_path = './mid_test/' + description
model_path = './model/' + description

restore = False
restore_point = 260000
Checkpoint = model_path + '/cVG iter ' + str(restore_point) + '/'

if not restore:
    restore_point = 0

saving_iter = 50000
Max_iter = 1000000

dPath = './List'

custom = CustomDataset('./tiny/')

data_loader = DataLoader(custom, batch_size=batch_size * GD_ratio, shuffle=True)

name_c, num_class = Ril.read_labeled_image_list(dPath + '/labels_tiny.txt')

generator = mm.Generator(n_noise, num_class).cuda()
discriminator = mm.Discriminator(num_class).cuda()

optim_disc = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.0, 0.9))
optim_gen = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.0, 0.9))

if restore:
    print('Weight Restoring.....')
    checkpoint = torch.load(Checkpoint)
    generator.load_state_dict(checkpoint['gen'])
    discriminator.load_state_dict(checkpoint['dis'])
    optim_gen.load_state_dict(checkpoint['opt_gen'])
    optim_disc.load_state_dict(checkpoint['opt_dis'])
    print('Weight Restoring Finish!')

start_time = time.time()
for e in range(10000):
    for iter_count, (img_real, class_img) in enumerate(data_loader):
        iter_count += restore_point

        for gd in range(GD_ratio):
            optim_disc.zero_grad()

            noise = torch.randn(batch_size, n_noise).cuda().detach()

            with torch.no_grad():
                img_gen = generator(noise, class_img[gd::GD_ratio]).detach()

            dis_fake = discriminator(img_gen, class_img[gd::GD_ratio])
            dis_real = discriminator(img_real[gd::GD_ratio], class_img[gd::GD_ratio])

            D_loss = torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. + dis_fake))
            D_loss.backward()
            optim_disc.step()

            if gd == 0:
                optim_gen.zero_grad()

                img_gen = generator(noise, class_img[gd::GD_ratio])
                dis_fake = discriminator(img_gen, class_img[gd::GD_ratio])

                G_loss = -torch.mean(dis_fake)
                G_loss.backward()
                optim_gen.step()

        if iter_count % 100 == 0:
            consume_time = time.time() - start_time
            print('%d     Epoch : %d\t\tLoss_D = %.4f\t\tLoss_G = %.4f\t\ttime = %.4f' %
                  (iter_count, e, D_loss.item(), G_loss.item(), consume_time))
            start_time = time.time()
            Loss1 = 0
            Loss2 = 0

        if iter_count % saving_iter == 0 and iter_count != restore_point:

            print('SAVING MODEL')
            Temp = model_path + '/cVG iter %s/' % iter_count

            if not os.path.exists(Temp):
                os.makedirs(Temp)

            SaveName = Temp + 'Train_%s' % iter_count
            torch.save({
                'gen': generator.state_dict(),
                'dis': discriminator.state_dict(),
                'opt_gen': optim_gen.state_dict(),
                'opt_dis': optim_disc.state_dict(),
            }, SaveName)
            print('SAVING MODEL Finish')

            print('Test start')
            with torch.no_grad():
                for rr in range(3):
                    for iclass in range(num_class):
                        class_name = name_c[iclass].split()[2]
                        for isave in range(100):
                            noise_test = torch.randn(1, n_noise).cuda().detach()
                            cl_num = torch.ones(1).type(torch.long).cuda() * iclass

                            img_sample = generator(noise_test, cl_num)

                            img_re = 255.0 * ((img_sample[0].cpu().numpy() + 1) / 2.0)
                            img_re = np.transpose(img_re, (1, 2, 0)).astype(np.uint8)

                            save_name = save_path + '/%d/%d' % (iter_count, rr)
                            name = save_name + '/img_%04d_%s.png' % (isave, class_name)

                            if not os.path.exists(save_name):
                                os.makedirs(save_name)
                            cv2.imwrite(name, img_re)

                print('Test Finish')
