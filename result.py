import torch
import time
import os
import numpy as np
from datasets import CustomDataset
from torch.utils.data import DataLoader
import torchvision
import module as mm
import fid
import inception_score

Height = 128
Width = 128
n_noise = 128

description = 'cBN'

save_path = './results/' + description
model_path = './model/' + description

saving_iter = 20000
Max_iter = 1000000

def gen_images(gen, n_noise, num_class, name_c, num_img=10000, batch=10, save_path='./mid_test/img', save_img=True):
    with torch.no_grad():
        gen.eval()

        npc = num_img//num_class//batch

        fake_images = []
        for i in range(num_img//batch):
            class_idx = i//npc
            class_name = name_c[class_idx].split()[2]

            label = torch.Tensor([class_idx]).cuda().long().repeat(batch)

            fake = gen(torch.randn(batch, n_noise).cuda(), label)

            fake_images.append(fake.cpu().numpy())

            # Save generated images.
            if save_img:
                isave = i % npc
                save_name = save_path + 'img_%s_%04d.png' % (class_name, isave)

                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torchvision.utils.save_image(fake[0], save_name, normalize=True)

        gen.train()
        return np.concatenate(fake_images)

def main():
    custom = CustomDataset('D:/dataset/tiny/')
    name_c = custom.label_name
    num_class = custom.num_label

    num_img = 50000

    generator = mm.Generator(n_noise, num_class).cuda()

    eval_data = iter(DataLoader(custom, batch_size=num_img))
    real_images = next(eval_data)[0].cpu().numpy()

    start_time = time.time()
    for iter_count in range(saving_iter, Max_iter+1, saving_iter):

        Checkpoint = model_path + '/cVG iter ' + str(iter_count) + '/Train_' + str(iter_count) + '.pth'

        print(iter_count)
        print('Weight Restoring.....')
        checkpoint = torch.load(Checkpoint)
        generator.load_state_dict(checkpoint['gen'])
        print('Weight Restoring Finish!')

        print('Evaluation start')

        fake_images = gen_images(generator, n_noise, num_class, name_c,
                                 save_path=save_path+'/%d/' % iter_count,
                                 num_img=num_img, save_img=False)

        # Calculate FID scores
        print('Calculate FID scores')
        fid_score = fid.calculate_fid(fake_images, real_images, batch_size=10)

        # Calculate Inception scores
        print('Calculate Inception scores')
        is_score = inception_score.inception_score(fake_images, batch_size=10, splits=10)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        with open(save_path + '/log_FID.txt', 'a+') as f:
            data = 'itr : %05d\t%.3f\n' % (
            iter_count, fid_score)
            f.write(data)
        with open(save_path + '/log_IS.txt', 'a+') as f:
            data = 'itr : %05d\t%.3f\t%.3f\n' % (
            iter_count, is_score[0], is_score[1])
            f.write(data)

        print('Evaluation Finish')

        consume_time = time.time() - start_time
        print(consume_time)
        start_time = time.time()

if __name__ == '__main__':
    main()