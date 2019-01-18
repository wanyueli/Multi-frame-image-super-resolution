#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import get_imgs_fn,  normalization_fn, image_shuffle
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
logdir = config.VALID.logdir

#ni = int(np.sqrt(batch_size))
ni = int(batch_size//2)

def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        #b_imgs=np.expand_dims(b_imgs,axis=-1)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    ## Train set
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr0_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr0_img_path, regx='.*.png', printable=False))
    train_lr1_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr1_img_path, regx='.*.png', printable=False))
    train_lr2_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr2_img_path, regx='.*.png', printable=False))
    train_lr3_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr3_img_path, regx='.*.png', printable=False))
    train_lr4_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr4_img_path, regx='.*.png', printable=False))
    ## Valid set
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr0_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr0_img_path, regx='.*.png', printable=False))
    valid_lr1_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr1_img_path, regx='.*.png', printable=False))
    valid_lr2_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr2_img_path, regx='.*.png', printable=False))
    valid_lr3_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr3_img_path, regx='.*.png', printable=False))
    valid_lr4_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr4_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    #train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    #train_lr0_imgs = read_all_imgs(train_lr0_img_list, path=config.TRAIN.lr0_img_path, n_threads=32)
    #train_lr1_imgs = read_all_imgs(train_lr1_img_list, path=config.TRAIN.lr1_img_path, n_threads=32)
    #train_lr2_imgs = read_all_imgs(train_lr2_img_list, path=config.TRAIN.lr2_img_path, n_threads=32)
    #train_lr3_imgs = read_all_imgs(train_lr3_img_list, path=config.TRAIN.lr3_img_path, n_threads=32)
    #train_lr4_imgs = read_all_imgs(train_lr4_img_list, path=config.TRAIN.lr4_img_path, n_threads=32)
    
    ## Shuffle images
    img_lr0_shuffle_list,img_lr1_shuffle_list,img_lr2_shuffle_list,img_lr3_shuffle_list,img_lr4_shuffle_list,img_hr_shuffle_list=image_shuffle(train_lr0_img_list,train_lr1_img_list,train_lr2_img_list,train_lr3_img_list,train_lr4_img_list,train_hr_img_list)
    
    train_hr_imgs = read_all_imgs(img_hr_shuffle_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr0_imgs = read_all_imgs(img_lr0_shuffle_list, path=config.TRAIN.lr0_img_path, n_threads=32)
    train_lr1_imgs = read_all_imgs(img_lr1_shuffle_list, path=config.TRAIN.lr1_img_path, n_threads=32)
    train_lr2_imgs = read_all_imgs(img_lr2_shuffle_list, path=config.TRAIN.lr2_img_path, n_threads=32)
    train_lr3_imgs = read_all_imgs(img_lr3_shuffle_list, path=config.TRAIN.lr3_img_path, n_threads=32)
    train_lr4_imgs = read_all_imgs(img_lr4_shuffle_list, path=config.TRAIN.lr4_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    t_image0 = tf.placeholder('float32', [batch_size, 128, 128, 3], name='t_image0_input_to_SRGAN_generator')
    t_image1 = tf.placeholder('float32', [batch_size, 128, 128, 3], name='t_image1_input_to_SRGAN_generator')
    t_image2 = tf.placeholder('float32', [batch_size, 128, 128, 3], name='t_image2_input_to_SRGAN_generator')
    t_image3 = tf.placeholder('float32', [batch_size, 128, 128, 3], name='t_image3_input_to_SRGAN_generator')
    t_image4 = tf.placeholder('float32', [batch_size, 128, 128, 3], name='t_image4_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 512, 512, 3], name='t_target_image')

    net_g = SRGAN_g(t_image0, t_image1, t_image2, t_image3, t_image4, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_d.print_params(False)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(t_image0, t_image1, t_image2, t_image3, t_image4, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    # d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    # d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')

    # d_loss1 = tl.cost.cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    # d_loss2 = tl.cost.cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    #
    # d_loss = d_loss1 + d_loss2

    #Wasserstein GAN Loss
    with tf.name_scope('w_loss/WARS_1'):
        d_loss =  tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
        tf.summary.scalar('w_loss', d_loss)

    #merged = tf.summary.merge_all()
    # loss_writer = tf.summary.FileWriter('/home/ubuntu/huzhihao/WARS/log/', sess.graph)
    # g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    g_gan_loss = - 1e-3 * tf.reduce_mean(logits_fake)
    #mse_loss = tl.cost.mean_squared_error(net_g.outputs , t_target_image, is_mean=True)
    mse_loss= tf.reduce_mean(tf.losses.absolute_difference(t_target_image,net_g.outputs))
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss +g_gan_loss
    #g_loss = mse_loss + vgg_loss
    
    tf.summary.scalar('g_loss', g_loss)
    merged = tf.summary.merge_all()
    
    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    #g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    g_optim_init = tf.train.RMSPropOptimizer(lr_v).minimize(mse_loss, var_list=g_vars)

    ## SRGAN
    # g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    g_optim = tf.train.RMSPropOptimizer(lr_v).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.RMSPropOptimizer(lr_v).minimize(d_loss, var_list=d_vars)

    # clip op
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]


    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    loss_writer = tf.summary.FileWriter(logdir, sess.graph)
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted( npz.items() ):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    sample_imgs0 = train_lr0_imgs[0:batch_size]
    sample_imgs1 = train_lr1_imgs[0:batch_size]
    sample_imgs2 = train_lr2_imgs[0:batch_size]
    sample_imgs3 = train_lr3_imgs[0:batch_size]
    sample_imgs4 = train_lr4_imgs[0:batch_size]
    # sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=normalization_fn)
    print('sample HR sub-image:',sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs0_96 = tl.prepro.threading_data(sample_imgs0, fn=normalization_fn)
    sample_imgs1_96 = tl.prepro.threading_data(sample_imgs1, fn=normalization_fn)
    sample_imgs2_96 = tl.prepro.threading_data(sample_imgs2, fn=normalization_fn)
    sample_imgs3_96 = tl.prepro.threading_data(sample_imgs3, fn=normalization_fn)
    sample_imgs4_96 = tl.prepro.threading_data(sample_imgs4, fn=normalization_fn)
    #sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    #print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    #tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit+'/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni//2, ni], save_dir_ginit+'/_train_sample_384.png')
    #tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan+'/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni//2, ni], save_dir_gan+'/_train_sample_384.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init+1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx : idx + batch_size],fn=normalization_fn)
            b0_imgs_96 = tl.prepro.threading_data(train_lr0_imgs[idx:idx+batch_size],fn=normalization_fn)
            b1_imgs_96 = tl.prepro.threading_data(train_lr1_imgs[idx:idx+batch_size],fn=normalization_fn)
            b2_imgs_96 = tl.prepro.threading_data(train_lr2_imgs[idx:idx+batch_size],fn=normalization_fn)
            b3_imgs_96 = tl.prepro.threading_data(train_lr3_imgs[idx:idx+batch_size],fn=normalization_fn)
            b4_imgs_96 = tl.prepro.threading_data(train_lr4_imgs[idx:idx+batch_size],fn=normalization_fn)
            
            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image0: b0_imgs_96, t_image1: b1_imgs_96, 
                               t_image2: b2_imgs_96, t_image3: b3_imgs_96, t_image4: b4_imgs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image0: sample_imgs0_96,t_image1: sample_imgs1_96,t_image2: sample_imgs2_96,
                                                t_image3: sample_imgs3_96,t_image4:sample_imgs4_96})
            print("[*] save images")
            tl.vis.save_images(out, [ni//2, ni], save_dir_ginit+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

    ###========================= train GAN (SRGAN) =========================###

    # clipping method
    # clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
    #                                      var in self.discriminator_variables]

    for epoch in range(0, n_epoch+1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0
        #total_g_loss, n_iter = 0, 0
        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx : idx + batch_size],fn=normalization_fn)
            b0_imgs_96 = tl.prepro.threading_data(train_lr0_imgs[idx:idx+batch_size],fn=normalization_fn)
            b1_imgs_96 = tl.prepro.threading_data(train_lr1_imgs[idx:idx+batch_size],fn=normalization_fn)
            b2_imgs_96 = tl.prepro.threading_data(train_lr2_imgs[idx:idx+batch_size],fn=normalization_fn)
            b3_imgs_96 = tl.prepro.threading_data(train_lr3_imgs[idx:idx+batch_size],fn=normalization_fn)
            b4_imgs_96 = tl.prepro.threading_data(train_lr4_imgs[idx:idx+batch_size],fn=normalization_fn)
            
            ## update D
            errD,summary, _, _ = sess.run([d_loss, merged,d_optim, clip_D], {t_image0:b0_imgs_96, t_image1:b1_imgs_96, 
                                        t_image2:b2_imgs_96, t_image3:b3_imgs_96, t_image4:b4_imgs_96,t_target_image: b_imgs_384})
            #errD, _, _ = sess.run([d_loss, d_optim, clip_D], {t_image0:b0_imgs_96, t_image1:b1_imgs_96, 
                                        #t_image2:b2_imgs_96, t_image3:b3_imgs_96, t_image4:b4_imgs_96,t_target_image: b_imgs_384})
            errG, summary, _ = sess.run([g_loss, merged, g_optim], {t_image0:b0_imgs_96, t_image1:b1_imgs_96, 
                                        t_image2:b2_imgs_96, t_image3:b3_imgs_96, t_image4:b4_imgs_96,t_target_image: b_imgs_384})
            loss_writer.add_summary(summary, epoch)
            # d_vars = sess.run(clip_discriminator_var_op)
            ## update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim],{t_image0:b0_imgs_96, t_image1:b1_imgs_96,
                                           t_image2:b2_imgs_96, t_image3:b3_imgs_96, t_image4:b4_imgs_96, t_target_image: b_imgs_384})

            print("Epoch [%2d/%2d] %4d time: %4.4fs, W_loss: %.8f, g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)"
                  % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image0:sample_imgs0_96, t_image1:sample_imgs1_96, t_image2:sample_imgs2_96, 
                                                t_image3:sample_imgs3_96, t_image4:sample_imgs4_96})
            print("[*] save images")
            tl.vis.save_images(out, [ni//2, ni], save_dir_gan+'/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)

def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.jpg', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.jpg', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr0_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr0_img_path, regx='.*.png', printable=False))
    valid_lr1_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr1_img_path, regx='.*.png', printable=False))
    valid_lr2_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr2_img_path, regx='.*.png', printable=False))
    valid_lr3_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr3_img_path, regx='.*.png', printable=False))
    valid_lr4_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr4_img_path, regx='.*.png', printable=False))
    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr0_imgs = read_all_imgs(valid_lr0_img_list, path=config.VALID.lr0_img_path, n_threads=32)
    valid_lr1_imgs = read_all_imgs(valid_lr1_img_list, path=config.VALID.lr1_img_path, n_threads=32)
    valid_lr2_imgs = read_all_imgs(valid_lr2_img_list, path=config.VALID.lr2_img_path, n_threads=32)
    valid_lr3_imgs = read_all_imgs(valid_lr3_img_list, path=config.VALID.lr3_img_path, n_threads=32)
    valid_lr4_imgs = read_all_imgs(valid_lr4_img_list, path=config.VALID.lr4_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    imid = 0 # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    #for imid in range(0, len(valid_hr_imgs)):
    valid_lr0_img = valid_lr0_imgs[imid]
    valid_lr1_img = valid_lr1_imgs[imid]
    valid_lr2_img = valid_lr2_imgs[imid]
    valid_lr3_img = valid_lr3_imgs[imid]
    valid_lr4_img = valid_lr4_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    #img_name = '0010_80.jpg'
    #valid_lr_img = get_imgs_fn(img_name, '/home/ubuntu/dataset/sr_test/testing/')  # if you want to test your own image
    valid_lr0_img = (valid_lr0_img / 127.5) - 1   # rescale to ［－1, 1]
    valid_lr1_img = (valid_lr1_img / 127.5) - 1
    valid_lr2_img = (valid_lr2_img / 127.5) - 1
    valid_lr3_img = (valid_lr3_img / 127.5) - 1
    valid_lr4_img = (valid_lr4_img / 127.5) - 1
    valid_hr_img=(valid_hr_img/127.5) - 1
    # print(valid_lr_img.min(), valid_lr_img.max())

    size0 = valid_lr0_img.shape
    size1 = valid_lr1_img.shape
    size2 = valid_lr2_img.shape
    size3 = valid_lr3_img.shape
    size4 = valid_lr4_img.shape
    t0_image = tf.placeholder('float32', [None, size0[0], size0[1], size0[2]], name='input_image0')
    t1_image = tf.placeholder('float32', [None, size1[0], size1[1], size1[2]], name='input_image1')
    t2_image = tf.placeholder('float32', [None, size2[0], size2[1], size2[2]], name='input_image2')
    t3_image = tf.placeholder('float32', [None, size3[0], size3[1], size3[2]], name='input_image3')
    t4_image = tf.placeholder('float32', [None, size4[0], size4[1], size4[2]], name='input_image4')
    
    # t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t0_image, t1_image, t2_image, t3_image, t4_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t0_image: [valid_lr0_img],t1_image: [valid_lr1_img],t2_image: [valid_lr2_img],
                                   t3_image: [valid_lr3_img],t4_image: [valid_lr4_img] })
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size2, out.shape)) # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    #tl.vis.save_image(out[0], save_dir+ '/gen_' + img_name[:-4] + '.png')
    tl.vis.save_image(out[0], save_dir + '/%svalid_gsr.png' %imid)
    #tl.vis.save_image(valid_lr_img, save_dir+'/valid_lr.png')
    #tl.vis.save_image(valid_hr_img, save_dir+'/valid_hr.png')
    
    #out_bicu = scipy.misc.imresize(valid_lr2_img, [size2[0]*4, size2[1]*4], interp='bicubic', mode=None)
    
    #tl.vis.save_image(out_bicu, save_dir + '/%svalid_bicubic.png'%imid)   
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
