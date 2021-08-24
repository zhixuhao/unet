from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import skimage.io as io
import skimage.transform as trans
import cv2
import os

import tensorflow as tf
print(tf.__version__)

#rgb
any                 = [192, 192, 192]   #wtite-gray
borders             = [0,0,255]         #blue
mitochondria        = [0,255,0]         #green
mitochondria_borders= [255,0,255]       #violet
PSD                 = [192,255,64]      #yellow
axon                = [255,128,64]      #yellow
vesicles            = [255,0,0]         #read

COLOR_DICT = np.array([mitochondria, PSD, vesicles,  axon, borders, mitochondria_borders])

def is_img(name):
    img_type = ('.png', '.jpg', '.jpeg')
    if name.endswith((img_type)):
        return True
    else:
        return False

def read_name_list(input_derectory, delete_name = None):
    dir_name_list = sorted(os.listdir(input_derectory), key=len)
    img_name_list = []
    for img_name in dir_name_list:
        if not is_img(img_name):
           continue
        else:
            if delete_name is not None:
                img_name_list.append(img_name.replace(delete_name,""))
            else:
                img_name_list.append(img_name)

    print("Count_pic:", len(img_name_list))
    return img_name_list

def get_normalise_data(data, target_size = (256,256), normalase_img_mod = "div255", normalase_mask_mode = "0_1", label_mask = True, num_class = 2):
    (img, mask) = data
    if normalase_img_mod == "linearTension01":
        max_img = img.max()
        min_img = img.min()
        img = (img-min_img)/(max_img-min_img)
        print("img min = " , min_img , " img max = ",max_img)
    elif normalase_img_mod == "div255":
        img = img/255.0

    if label_mask:
        mod_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            mod_mask[mask == i,i]=1
        mask = mod_mask

    img = trans.resize(img, target_size)
    mask = cv2.resize(mask, target_size)

    if normalase_mask_mode == "to_0_1":
        mask[mask < 128] = 0
        mask[mask > 127] = 1
    elif normalase_mask_mode == "to_-1_1":
         mask[mask < 128] = -1
         mask[mask > 127] = 1

    return (img,mask)

def load_data(dir_img_name, dir_mask_name, normalise_data_parameters, delete_mask_name = None, color_mode_img = "gray", color_mode_mask = "gray"):
    (target_size, normalase_img_mod, normalase_mask_mode, label_mask, num_class) = normalise_data_parameters

    img_name_list = set(read_name_list(dir_img_name))
    mask_name_list = set(read_name_list(dir_mask_name, delete_mask_name))
    cross = img_name_list & mask_name_list
    print("crossing img and mask: ", len(cross))

    X = []
    Y = []

    for img_and_mask_name in cross:
        img_name_path = os.path.join(dir_img_name, img_and_mask_name)
        mask_name_path = os.path.join(dir_mask_name, img_and_mask_name)
        if color_mode_img == "gray":
            img = cv2.imread(img_name_path,0)
        else:
            img = cv2.imread(img_name_path)

        if color_mode_mask == "gray":
            mask = cv2.imread(mask_name_path, 0)
        else:
            mask = cv2.imread(mask_name_path)

        (img, mask) = get_normalise_data((img, mask), target_size, normalase_img_mod, normalase_mask_mode, label_mask, num_class)
        X.append(img)
        Y.append(mask)

 #       yield (img,mask
    return X,Y

def load_data_multi_mask(dir_img_name, dir_mask_name, normalise_data_parameters, mask_name_label_list = [], delete_mask_name = None, color_mode_img = "gray", color_mode_mask = "gray"):
    (target_size, normalase_img_mod, normalase_mask_mode, label_mask, num_class) = normalise_data_parameters

    img_name_list = set(read_name_list(dir_img_name))
    cross = img_name_list
    for name_label in mask_name_label_list[0:num_class]:
        set_mask = set(read_name_list(os.path.join(dir_mask_name, name_label), delete_mask_name))
        cross = cross & set_mask
    print("crossing img and mask: ", len(cross))

    X = []
    Y = []

    #cross = sorted(list(cross))
    for img_and_mask_name in cross:
        #print(img_and_mask_name)
        img_name_path = os.path.join(dir_img_name, img_and_mask_name)
        if color_mode_img == "gray":
            img = cv2.imread(img_name_path,0)
            img = np.expand_dims(img, axis=-1)
        else:
            img = cv2.imread(img_name_path)

        mask_name_path = None
        mask_list = np.zeros((img.shape[0:2] + (num_class,)), np.float32)
        for i,name_label in enumerate(mask_name_label_list[0:num_class]):
            mask_name_path = os.path.join(dir_mask_name, name_label, img_and_mask_name)
            if color_mode_mask == "gray":
                mask = cv2.imread(mask_name_path, 0)
            else:
                mask = cv2.imread(mask_name_path)
            mask_list[:,:,i] = mask.astype(np.float32)
        #print(img_name_path)
        #print(mask_name_path)

        (img, mask) = get_normalise_data((img, mask_list), target_size, normalase_img_mod, normalase_mask_mode, label_mask, num_class)
        if num_class == 1:
            mask = np.expand_dims(mask, axis=-1)

        #temp_img = img
        #temp_mask = mask * 255

        ##cv2.imshow("test X", temp_img[:,:,0])
        #cv2.imshow("test Y", temp_mask)
        #cv2.waitKey()
        X.append(img)
        Y.append(mask)

    #       yield (img,mask
    X = np.asarray(X)
    Y = np.asarray(Y)

    print(X.shape)
    print(Y.shape)

    #for i in range(len(cross)):
    #    cv2.imshow("test X", X[i])
    #    cv2.imshow("test Y", Y[i])
    #    io.imsave(str(i) + " test Y.png", Y[i])
    #    cv2.waitKey()

    return X, Y

def get_train_generator_data(dir_img_name,
                             dir_mask_name,
                             aug_dict,
                             batch_size,
                             list_name_label_mask = None,
                             delete_mask_name = None,
                             target_size = (256,256),
                             color_mode_img = "gray",
                             color_mode_mask = "gray",
                             normalase_img_mod = "div255",
                             num_class = 2,
                             label_mask = False,
                             normalase_mask_mode = None,
                             save_prefix_image = "image",
                             save_prefix_mask = "mask",
                             save_to_dir = None,
                             seed = 1,
                             shuffle = False
                             ):
    normalise_data_parameters = (target_size, normalase_img_mod, normalase_mask_mode, label_mask, num_class)

    datagen = ImageDataGenerator(**aug_dict)
    if list_name_label_mask is None:
        X,Y = load_data(dir_img_name, dir_mask_name,
                        normalise_data_parameters,
                        delete_mask_name,
                        color_mode_img, color_mode_mask)
    else:
        X, Y = load_data_multi_mask(dir_img_name, dir_mask_name,
                                    normalise_data_parameters,
                                    list_name_label_mask,
                                    delete_mask_name,
                                    color_mode_img, color_mode_mask)


    genX1 = datagen.flow(X, Y, batch_size=batch_size, seed=seed, save_prefix = save_prefix_image, save_to_dir = save_to_dir)
    genX2 = datagen.flow(Y, X, batch_size=batch_size, seed=seed, save_prefix = save_prefix_mask, save_to_dir = save_to_dir)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()

        yield X1i[0], X2i[0]


def testGenerator(test_path, name_list = [], save_dir = None, num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    img_type = ('.png', '.jpg', '.jpeg')
    name_dir_list = [img_name for img_name in sorted(os.listdir(test_path), key=len) if img_name.endswith(img_type)]

    for img_name in name_dir_list[0:num_image]:
        name_list.append(img_name)
        img = io.imread(os.path.join(test_path, img_name),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        if save_dir is not None:
            io.imsave(os.path.join(save_dir, img_name), img[0])
        yield img

def labelVisualize(num_class, trust_percentage, color_dict,img):
    if num_class == 1:
        return img
    else:
        #print(img.shape)
        #for i in range(num_class):
        #    print(str(i)+":",  img[:, :, i].max(), " ",img[:, :, i].min() )

        img_out = np.zeros(img.shape[0:2] + (3,))
    #    print(img_out.shape)
        for i in range(num_class):
            img_out[img[:,:,i] >= trust_percentage] = color_dict[i]
        return img_out/255

def saveResult(save_path,npyfile, namelist, trust_percentage = 0.9 ,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class, trust_percentage, COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"predict_"+namelist[i]),img)


mask_name_label_list = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

def viewResult(save_path,npyfile, namelist, trust_percentage = 0.9 ,flag_multi_class = False,num_class = 2):
    for i, item in enumerate(npyfile):
        for n_class in range(num_class):
            cv2.imshow(mask_name_label_list[n_class]+ "_"+ namelist[i], item[:,:,n_class])
        cv2.waitKey()
        cv2.destroyAllWindows()


def saveResultMask(save_path,npyfile, namelist,num_class = 2):
    for i,item in enumerate(npyfile):
        for class_index in range(num_class):
            out_dir = os.path.join(save_path, mask_name_label_list[class_index])
            if not os.path.isdir(out_dir):
                print("создаю out_dir:" + out_dir)
                os.makedirs(out_dir)

            if (os.path.isfile(os.path.join(out_dir, "predict_" + namelist[i]))):
                os.remove(os.path.join(out_dir, "predict_" + namelist[i]))
            io.imsave(os.path.join(out_dir, "predict_" + namelist[i]), item[:,:,class_index])
