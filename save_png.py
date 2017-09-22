'''
    tranfer the test dataset output to imgs
'''
# coding:UTF-8
from keras.preprocessing.image import array_to_img
import numpy as np


def save_output(default_dir):
    '''
    save test dataset into imgs into a default dir
    '''

    imgs_train = np.load(file= "./imgs_mask_test.npy")
    imgs_train = imgs_train.astype('float32')

    pictures = np.shape(imgs_train)[0]

    for i in range(pictures):
        img_tmp = array_to_img(imgs_train[i])
        img_tmp.save(default_dir+"{}.".format(i)+"png")

if __name__ == "__main__":
    default_dir = "./aoao/"
    save_output(default_dir)