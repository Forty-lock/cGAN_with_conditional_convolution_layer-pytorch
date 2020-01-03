import numpy as np
import os
import random
import cv2

def make_list(dPath):

    file_list = os.listdir(dPath)
    file_list.sort()
    # random.shuffle(file_list)

    for k in range(len(file_list)):

        list_name = dPath + '/File_List.txt'

        f = open(list_name, 'a+')
        data = dPath + ('/%s\n' % (file_list[k]))

        f.write(data)
        f.close()
        if k%100 == 0:
            print('%d / %d' % (k, len(file_list)))

def read_labeled_image_list(image_list_file):
    #"""Reads a .txt file containing pathes and labeles
    #Args:
    #   image_list_file: a .txt file with one /path/to/image per line
    #   label: optionally, if set label will be pasted after each line
    #Returns:
    #   List with all filenames in file image_list_file

    f = open(image_list_file, 'r')
    filenames1 = []
    Total_Image_Num = 0

    for line in f:
        filename1 = line[:-1]

        filenames1.append(filename1)

        Total_Image_Num = Total_Image_Num + 1

    return filenames1, Total_Image_Num

def read_labeled_image_list2(image_list_file):
    #"""Reads a .txt file containing pathes and labeles
    #Args:
    #   image_list_file: a .txt file with one /path/to/image per line
    #   label: optionally, if set label will be pasted after each line
    #Returns:
    #   List with all filenames in file image_list_file

    f = open(image_list_file, 'r')
    filenames1 = []
    Total_Image_Num = 0

    for line in f:
        filename1 = line[:-1].split()[0]

        filenames1.append(filename1)

        Total_Image_Num = Total_Image_Num + 1

    return filenames1, Total_Image_Num

def MakeImageBlock_pickle(pickle_image, pickle_class, Height, Width, i, batch_size, resize=True):

    iCount = 0
    Image = np.zeros((batch_size, 3, Height, Width))
    Class = np.zeros((batch_size))

    #Query Image block

    for iL in range((i * batch_size), (i * batch_size) + batch_size):

        Loadimage = pickle_image[iL]
        Class[iCount] = int(pickle_class[iL]) - 1

        # if Gray make it colors
        if Loadimage.ndim == 2:
            Loadimage = np.expand_dims(Loadimage, 2)
            Loadimage = np.tile(Loadimage, (1, 1, 3))

        if Loadimage.shape[2] is not 3:
            Loadimage = Loadimage[:, :, 0:3]

        if resize:
            Loadimage = cv2.imresize(Loadimage, (Width, Height))

        Loadimage = Loadimage.astype(np.float32)

        # Mean Value subtraction
        Loadimage = (Loadimage / 255.0 - 0.5) * 2

        Loadimage = np.transpose(Loadimage, (2, 0, 1))

        Image[iCount] = Loadimage
        iCount += 1

    return Image, Class
