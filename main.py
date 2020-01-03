import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable
import module as mm
import os
import pickle
import numpy as np
import Read_Image_List as Ril
from operator import itemgetter
import cv2

Height = 128
Width = 128
batch_size = 16
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

saving_iter = 5000
Max_iter = 300000

dPath = './List'

with open('D:/dataset/tiny/tiny_ImageNet128.pkl', 'rb') as pickle_file:
    pickle_image = pickle.load(pickle_file)
    num_f = len(pickle_image)
with open('D:/dataset/tiny/tiny_ImageNet128_class.pkl', 'rb') as pickle_file:
    pickle_class = pickle.load(pickle_file)

order_pickle = np.arange(num_f)

name_c, num_class = Ril.read_labeled_image_list(dPath + '/labels_tiny.txt')
total_batch = num_f // batch_size

generator = mm.Generator(n_noise, num_class).cuda()
discriminator = mm.Discriminator().cuda()

if restore:
    print('Weight Restoring.....')
    checkpoint = torch.load(Checkpoint)
    generator.load_state_dict(checkpoint['gen'])
    discriminator.load_state_dict(checkpoint['dis'])
    print('Weight Restoring Finish!')

optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=0.0002, betas=(0.0,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.0,0.9))

start_time = time.time()
for iter_count in range(restore_point, Max_iter+1):

    i = iter_count % (total_batch // GD_ratio)
    e = iter_count // total_batch

    if i == 0:
        np.random.shuffle(order_pickle)
        pickle_image = itemgetter(*order_pickle)(pickle_image)
        pickle_class = itemgetter(*order_pickle)(pickle_class)

    for c in range(GD_ratio):
        img_real, class_img = Ril.MakeImageBlock_pickle(pickle_image, pickle_class, Height, Width, i * GD_ratio + c, batch_size, resize=False)
        noise = Variable(torch.randn(batch_size, n_noise).cuda())
        img_real = torch.from_numpy(img_real).type(torch.float).cuda()
        class_img = torch.from_numpy(class_img).type(torch.long).cuda()

        optim_gen.zero_grad()
        optim_disc.zero_grad()

        D_loss = nn.ReLU()(1.0 - discriminator(img_real)).mean() + nn.ReLU()(1.0 + discriminator(generator(noise, class_img))).mean()
        D_loss.backward()
        optim_disc.step()

    optim_gen.zero_grad()
    optim_disc.zero_grad()

    noise = Variable(torch.randn(batch_size, n_noise).cuda())
    G_loss = -discriminator(generator(noise, class_img)).mean()
    G_loss.backward()
    optim_gen.step()

    if iter_count % 100 == 0:
        consume_time = time.time() - start_time
        print('%d     Epoch : %d       D Loss = %.5f       G Loss = %.5f    time = %.4f' % (iter_count, e, D_loss, G_loss, consume_time))
        start_time = time.time()

    if iter_count % saving_iter == 0 and iter_count != restore_point:

        print('SAVING MODEL')
        Temp = model_path + '/cVG iter %s/' % iter_count

        if not os.path.exists(Temp):
            os.makedirs(Temp)

        SaveName = Temp + 'Train_%s' % iter_count
        torch.save({
            'gen': generator.state_dict(),
            'dis': discriminator.state_dict(),
        }, SaveName)
        print('SAVING MODEL Finish')

        print('Test start')

        for rr in range(3):
            for iclass in range(num_class):
                class_name = name_c[iclass].split()[2]
                for isave in range(100):
                    noise = Variable(torch.randn(1, n_noise).cuda())
                    cl_num = Variable(torch.zeros(1).type(torch.long).cuda()) * iclass

                    img_sample = generator(noise, cl_num)

                    img_re = 255.0 * ((img_sample[0] + 1) / 2.0)

                    save_name = save_path + '/%d/%d' % (iter_count, rr)
                    name = save_name + '/img_%04d_%s.png' % (isave, class_name)

                    if not os.path.exists(save_name):
                        os.makedirs(save_name)
                    cv2.imwrite(name, img_re)

        print('Test Finish')
