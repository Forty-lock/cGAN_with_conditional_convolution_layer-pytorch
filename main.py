import torch
import torchvision
import torch.optim as optim
import time
import module as mm
import os
import numpy as np
from datasets import CustomDataset
from torch.utils.data import DataLoader
from evaluation import evaluate

Height = 128
Width = 128
batch_size = 32
n_noise = 128

GD_ratio = 5

description = 'cConv'

save_path = './mid_test/' + description
model_path = './model/' + description

restore = False
restore_point = 380000
Checkpoint = model_path + '/cVG iter ' + str(restore_point) + '/Train_' + str(restore_point) + '.pth'

if not restore:
    restore_point = 0

saving_iter = 10000
Max_iter = 1000000

def save_images(gen, n_noise, num_class, name_c, npc=10, save_path='./mid_test/img/'):
    with torch.no_grad():
        gen.eval()

        for i in range(npc*num_class):
            class_idx = i//npc
            class_name = name_c[class_idx].split()[2]

            label = torch.Tensor([class_idx]).cuda().long()

            fake = gen(torch.randn(1, n_noise).cuda(), label)

            # Save generated images.
            isave = i % npc
            save_name = save_path + 'img_%s_%04d.png' % (class_name, isave)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torchvision.utils.save_image(fake[0], save_name, normalize=True)

        gen.train()

def main():
    custom = CustomDataset('D:/dataset/tiny/')
    name_c = custom.label_name
    num_class = custom.num_label

    data_loader = DataLoader(custom, batch_size=batch_size * GD_ratio, shuffle=True, drop_last=True)

    generator = mm.Generator(n_noise, num_class).cuda()
    discriminator = mm.Discriminator(num_class).cuda()

    optim_disc = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.0, 0.9))
    optim_gen = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.0, 0.9))

    if restore:
        print('Weight Restoring.....')
        generator.load_state_dict(torch.load(Checkpoint)['gen'])
        discriminator.load_state_dict(torch.load(Checkpoint)['dis'])
        optim_gen.load_state_dict(torch.load(Checkpoint)['opt_gen'])
        optim_disc.load_state_dict(torch.load(Checkpoint)['opt_dis'])
        torch.cuda.empty_cache()
        print('Weight Restoring Finish!')

    print('Training start')
    is_training = True
    iter_count = restore_point
    start_time = time.time()
    for e in range(100000):
        if not is_training:
            break
        for step, (img_real, class_img) in enumerate(data_loader):
            D_loss = 0

            for gd in range(GD_ratio):
                with torch.no_grad():
                    img_gen = generator(torch.randn(batch_size, n_noise).cuda(), class_img[gd::GD_ratio].cuda())

                dis_fake = discriminator(img_gen, class_img[gd::GD_ratio].cuda())
                dis_real = discriminator(img_real[gd::GD_ratio].cuda(), class_img[gd::GD_ratio].cuda())

                D_loss = torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. + dis_fake))
                optim_disc.zero_grad()
                D_loss.backward()
                optim_disc.step()

            img_gen = generator(torch.randn(batch_size, n_noise).cuda(), class_img[0::GD_ratio].cuda())
            dis_fake = discriminator(img_gen, class_img[0::GD_ratio].cuda())
            dis_real = None

            G_loss = -torch.mean(dis_fake)
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            G_loss.backward()
            optim_gen.step()
            iter_count += 1

            if iter_count % 100 == 0:
                consume_time = time.time() - start_time
                print('%d\t\tEpoch : %d\t\tLoss_D = %.3f\t\tLoss_G = %.3f\t\ttime = %.4f' %
                      (iter_count, e, D_loss.item(), G_loss.item(), consume_time))
                start_time = time.time()

            if iter_count % saving_iter == 0:

                print('SAVING MODEL')
                Temp = model_path + '/cVG iter %s/' % iter_count

                if not os.path.exists(Temp):
                    os.makedirs(Temp)

                SaveName = Temp + 'Train_%s.pth' % iter_count
                torch.save({
                    'gen': generator.state_dict(),
                    'dis': discriminator.state_dict(),
                    'opt_gen': optim_gen.state_dict(),
                    'opt_dis': optim_disc.state_dict(),
                }, SaveName)
                print('SAVING MODEL Finish')

                save_images(generator, n_noise, num_class, name_c, save_path=save_path + '/img/')

                # print('Evaluation start')
                #
                # fid_score, is_score = evaluate(generator, n_noise, num_class, name_c, custom,
                #                                num_img=50000, save_path=save_path + '/img/')
                #
                # with open(save_path + '/log_FID.txt', 'a+') as f:
                #     data = 'itr : %05d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % (
                #     iter_count, fid_score[0], fid_score[1], fid_score[2], np.average(fid_score), np.std(fid_score))
                #     f.write(data)
                # with open(save_path + '/log_IS.txt', 'a+') as f:
                #     data = 'itr : %05d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % (
                #     iter_count, is_score[0], is_score[1], is_score[2], np.average(is_score), np.std(is_score))
                #     f.write(data)
                # print('Evaluation Finish')

            if iter_count == Max_iter:
                is_training = False
                break

if __name__ == '__main__':
    main()