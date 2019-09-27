# Multi-frame-image-super-resolution
### Introduction of code file
*augmentation.py ---data augmentation
*defshuffle.py   ---shuffle images or image patches
*config.py       ---configuration hyperparameter

### Run
* Installation
'''
   pip install tensorlayer==1.8.0
   conda install tensorflow-gpu==1.8.0
   pip install easydict
'''
'''python
   config.VALID.img_path='your_image_folder/'
   config.TRAIN.img_path='your_image_folder/'
   config.VALID.logdir='your-tensorboard_folder/'
 '''
 * Start Training   ----- python main.py
 * Start Testing    ----- python main.py --mode=evaluate
 
 Reference
 SRGAN_Wasserstein https://github.com/JustinhoCHN/SRGAN_Wasserstein
 
PS：
if some parts of this project, please cite this paper "W. Li, Y. He, W. Kong, F. Gao, J. Wang, and G. Shi, "Enhancement of Retinal Image From Line-Scanning Ophthalmoscope Using Generative Adversarial Networks," IEEE Access 7, 99830–99841 (2019)."

 
