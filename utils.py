import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import crop, imresize
# from config import config, log_config
#
# img_path = config.TRAIN.img_path
from collections import defaultdict
import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def crop_sub_imgs_fn2(x, is_random=True):
    x = crop(x, wrg=96, hrg=96, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def normalization_fn(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def image_shuffle(train_img_list0,train_img_list1,train_img_list2,train_img_list3,train_img_list4,train_img_list5):
    d=defaultdict(list)
    for key in range(0,len(train_img_list0)):
        d[key].append(train_img_list0[key])

    k = list(d.keys())
    np.random.shuffle(k)
    img_shuffle_list0=[]
    img_shuffle_list1=[]
    img_shuffle_list2=[]
    img_shuffle_list3=[]
    img_shuffle_list4=[]
    img_shuffle_list5=[]

    for i in range(0,len(train_img_list0)):
        k1=k[i]
        list0=train_img_list0[k1]
        img_shuffle_list0.append(list0)
    
    for i in range(0,len(train_img_list1)):
        k2=k[i]
        list1=train_img_list1[k2]
        img_shuffle_list1.append(list1)
    
    for i in range(0,len(train_img_list2)):
        k3=k[i]
        list2=train_img_list2[k3]
        img_shuffle_list2.append(list2)
    
    for i in range(0,len(train_img_list3)):
        k4=k[i]
        list3=train_img_list3[k4]
        img_shuffle_list3.append(list3)
    
    for i in range(0,len(train_img_list4)):
        k5=k[i]
        list4=train_img_list4[k5]
        img_shuffle_list4.append(list4)

    for i in range(0,len(train_img_list5)):
        k1=k[i]
        list5=train_img_list5[k1]
        img_shuffle_list5.append(list5)
    
    return img_shuffle_list0,img_shuffle_list1,img_shuffle_list2,img_shuffle_list3,img_shuffle_list4,img_shuffle_list5