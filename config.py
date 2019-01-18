from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 8
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 0
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 150
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/train_HR/'
config.TRAIN.lr0_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/train_LR0/'
config.TRAIN.lr1_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/train_LR1/'
config.TRAIN.lr2_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/train_LR2/'
config.TRAIN.lr3_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/train_LR3/'
config.TRAIN.lr4_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/train_LR4/'
#config.TRAIN.hr_img_path = '/home/ubuntu/dataset/image_tag/srgan_all_jpg/trn_hr/'
#config.TRAIN.lr_img_path = '/home/ubuntu/dataset/image_tag/srgan_all_jpg/trn_lr/'


config.VALID = edict()
## test set location
config.VALID.hr_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/valid_HR/'
config.VALID.lr0_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/LR0/'
config.VALID.lr1_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/LR1/'
config.VALID.lr2_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/LR2/'
config.VALID.lr3_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/LR3/'
config.VALID.lr4_img_path = '/home/amax/Documents/my_code/VSRgan_test2/data/LR4/'


#config.VALID.hr_img_path = '/home/ubuntu/dataset/image_tag/srgan_all_jpg/val_hr/'
#config.VALID.lr_img_path = '/home/ubuntu/dataset/image_tag/srgan_all_jpg/val_lr/'

config.VALID.logdir = '/home/amax/Documents/my_code/VSRgan_test2/log/'
def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
