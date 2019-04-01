# sem-mask-nn
Neural network that generate semantic mask

Project 1 for CV course


1. March 23rd random crop 576*800, 
2. the loss functions to cross entropy loss
3. March 30th change FCN32 to FCN8, adjust batchsize to 8 and lr 8*10e-4, 
   try augmentation 
   
   分析：we test the result on FCN8,FCN32, Epoch 25/50, found the result is good 
   momentum = 0 的结果有时候大于 0.9 比如 最开始跑FCN32 batchsize1 的时候可以0.469 
   FCN 32 batch1 epoch 22 是0.4717 （第22个epoch, 23就直接掉到0.3953
   
   
4. if not good then change VGG to resnet 

