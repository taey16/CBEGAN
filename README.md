# CBEGAN
- A simple conditional version of the Boundary Equilibrium Generative Adversarial Networks (CBEGANs) 
- M. Mirza et. al, Conditional Generative Adversarial Nets, 2014 
- D. Berthelot et. al, BEGAN: Boundary Equilibrium Generative Adversarial Networks, 2017
- **We generate male or female face images with latent code (i.e. z) conditioned on gender-attribute vector.**

# data preperation
- [download CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): download Align&Croppped images, Attribute Annotations, and Train/Val/Test partitions
- Run [parseCelebA_gender_faceCrop.py](https://github.com/taey16/CBEGAN/blob/master/preprocess/parseCelebA_gender_faceCrop.py) (take care of specifying celebRawImgRoot and list_attr_celeba.data.txt)
- We found out **if we use celeba raw image without face-crop, the algorithm does not generate male or female images well.**

# Results
- Train with [main_CBEGAN.py](https://github.com/taey16/CBEGAN/blob/master/main_CBEGAN.py): You should check [meta parameters properly](https://github.com/taey16/CBEGAN/blob/master/main_CBEGAN.py#L23-L50) in training.
- ```CUDA_VISIBLE_DEVICES=x python main_CBEGAN.py --dataroot /path/to/CelebA/gender_facecrop/train --exp /path/to/dir/for/checkpoints --cond_size 2```
- Run [interpolateCond.py](https://github.com/taey16/CBEGAN/blob/master/interpolateCond.py): We generate z with uniform distribution (from -1 to 1) and then set condition vector to [1, -1] or [-1, 1]
- ```CUDA_VISIBLE_DEVICES=x python interpolateCond.py --exp /path/to/dir/for/saving/result --netG /path/to/your/netG_epoch_xx.pth```

![result](https://github.com/taey16/CBEGAN/blob/master/imgs/CBEGAN_celeb_gender.png)
