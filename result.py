import torch
import time
import module as mm
import os
import numpy as np
from datasets import CustomDataset
from evaluation import evaluate

Height = 128
Width = 128
n_noise = 128

description = 'cConv'

save_path = './results/' + description
model_path = './model/' + description

saving_iter = 20000
Max_iter = 1000000

def main():
    custom = CustomDataset('D:/dataset/tiny/')
    name_c = custom.label_name
    num_class = custom.num_label

    generator = mm.Generator(n_noise, num_class).cuda()

    start_time = time.time()
    for iter_count in range(saving_iter, Max_iter+1, saving_iter):

        Checkpoint = model_path + '/cVG iter ' + str(iter_count) + '/Train_' + str(iter_count) + '.pth'

        print(iter_count)
        print('Weight Restoring.....')
        checkpoint = torch.load(Checkpoint)
        generator.load_state_dict(checkpoint['gen'])
        print('Weight Restoring Finish!')

        print('Evaluation start')

        fid_score, is_score = evaluate(generator, n_noise, num_class, name_c, custom,
                                       num_img=50000, time=3, batch=50, save_img=False)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        with open(save_path + '/log_FID.txt', 'a+') as f:
            data = 'itr : %05d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % (
            iter_count, fid_score[0], fid_score[1], fid_score[2], np.average(fid_score), np.std(fid_score))
            f.write(data)
        with open(save_path + '/log_IS.txt', 'a+') as f:
            data = 'itr : %05d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % (
            iter_count, is_score[0], is_score[1], is_score[2], np.average(is_score), np.std(is_score))
            f.write(data)

        print('Evaluation Finish')

        consume_time = time.time() - start_time
        print(consume_time)
        start_time = time.time()

if __name__ == '__main__':
    main()