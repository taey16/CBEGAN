# CBEGAN
- A conditional version of the Boundary Equilibrium Generative Adversarial Networks (CBEGANs) 
- M. Mirza et. al, Conditional Generative Adversarial Nets, 2014 
- D. Berthelot et. al, BEGAN: Boundary Equilibrium Generative Adversarial Networks, 2017

# data preperation
- [download CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html): download Align&Croppped images, Attribute Annotations, and Train/Val/Test partitions
- Run [parseCelebA_gender_faceCrop.py](https://github.com/taey16/CBEGAN/blob/master/preprocess/parseCelebA_gender_faceCrop.py) (take care of specifying celebRawImgRoot and list_attr_celeba.data.txt)
- We found out **if we use celeba raw image without face-crop, the algorithm does not generate male of female images well.**
