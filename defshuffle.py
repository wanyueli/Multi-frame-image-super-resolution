#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:09:08 2018

@author: amax
"""
import os, time, pickle, random
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import copy
import tensorflow as tf
import tensorlayer as tl
#from SRResnet import SRGAN_g
from utils import get_imgs_fn
from config import config, log_config
from collections import defaultdict

config.TRAIN.hr_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/train_HR/'
config.TRAIN.lr0_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/LR0/'
config.TRAIN.lr1_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/LR1/'
config.TRAIN.lr2_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/LR2/'
config.TRAIN.lr3_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/LR3/'
config.TRAIN.lr4_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/LR4/'

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

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
    #d1=defaultdict(list)

    for i in range(0,len(train_img_list0)):
        k1=k[i]
        list0=train_img_list0[k1]
        img_shuffle_list0.append(list0)
    #img0=read_all_imgs(img_shuffle_list0, path=path0, n_threads=32)
    
    for i in range(0,len(train_img_list1)):
        k2=k[i]
        list1=train_img_list1[k2]
        img_shuffle_list1.append(list1)
    #img1=read_all_imgs(img_shuffle_list1, path=path1, n_threads=32)
    
    for i in range(0,len(train_img_list2)):
        k3=k[i]
        list2=train_img_list2[k3]
        img_shuffle_list2.append(list2)
    #img2=read_all_imgs(img_shuffle_list2, path=path2, n_threads=32)
    
    for i in range(0,len(train_img_list3)):
        k4=k[i]
        list3=train_img_list3[k4]
        img_shuffle_list3.append(list3)
    #img3=read_all_imgs(img_shuffle_list3, path=path3, n_threads=32)
    
    for i in range(0,len(train_img_list4)):
        k5=k[i]
        list4=train_img_list4[k5]
        img_shuffle_list4.append(list4)
    #img4=read_all_imgs(img_shuffle_list4, path=path4, n_threads=32)

    for i in range(0,len(train_img_list5)):
        k1=k[i]
        list5=train_img_list5[k1]
        img_shuffle_list5.append(list5)
    #img5=read_all_imgs(img_shuffle_list5, path=path5, n_threads=32)
    
    return img_shuffle_list0,img_shuffle_list1,img_shuffle_list2,img_shuffle_list3,img_shuffle_list4,img_shuffle_list5
    
train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
train_lr0_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr0_img_path, regx='.*.png', printable=False))
train_lr1_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr1_img_path, regx='.*.png', printable=False))
train_lr2_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr2_img_path, regx='.*.png', printable=False))
train_lr3_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr3_img_path, regx='.*.png', printable=False))
train_lr4_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr4_img_path, regx='.*.png', printable=False))
path0=config.TRAIN.hr_img_path
path1=config.TRAIN.lr0_img_path
path2=config.TRAIN.lr1_img_path
path3=config.TRAIN.lr2_img_path
path4=config.TRAIN.lr3_img_path
path5=config.TRAIN.lr4_img_path
#train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
"""
train_shufflehr_imgs,img_shufflehr_list= image_shuffle(train_hr_img_list,train_hr_img_list,path=config.TRAIN.hr_img_path)
train_shufflelr0_imgs,img_shufflelr0_list= image_shuffle(train_hr_img_list,train_lr0_img_list,path=config.TRAIN.lr0_img_path)
train_shufflelr1_imgs,img_shufflelr1_list= image_shuffle(train_hr_img_list,train_lr1_img_list,path=config.TRAIN.lr1_img_path)
train_shufflelr2_imgs,img_shufflelr2_list= image_shuffle(train_hr_img_list,train_lr2_img_list,path=config.TRAIN.lr2_img_path)
train_shufflelr3_imgs,img_shufflelr3_list= image_shuffle(train_hr_img_list,train_lr3_img_list,path=config.TRAIN.lr3_img_path)
train_shufflelr4_imgs,img_shufflelr4_list= image_shuffle(train_hr_img_list,train_lr4_img_list,path=config.TRAIN.lr4_img_path)
"""
img_shuffle_list0,img_shuffle_list1,img_shuffle_list2,img_shuffle_list3,img_shuffle_list4,img_shuffle_list5=image_shuffle(train_lr0_img_list,train_lr1_img_list,train_lr2_img_list,train_lr3_img_list,train_lr4_img_list,train_hr_img_list)