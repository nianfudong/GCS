# PyTorch-GCS
GCS loss for 2D facial landmark localization in PyTorch.

## Installation
    $ python 3.6
    $ PyTorch 0.4.1
    $ OpenCV 3.X
    
## Train
    $ python train.py 

mylambda in train.py (line 150) controls the hyper-parameter.

How to select different loss function:

    $ criterion = Wing_loss.Wing_loss()     # Wing Loss (Feng et al. CVPR2018)
    $ criterion = nn.MSELoss()              # General L2 loss
    $ criterion = nn.L1Loss()               # General L1 loss
    $ criteriongcs = Gcs_loss.Gcs_loss()    # The proposed GCS loss
    
## Test
    $ python netforward.py
   
## Eval
    $ python evaluation.py
    
## Label Format
    Input Label file: imagename point_1.x point_1.y point_2.x point_2.y point_3.x point_3.y point_4.x point_4.y ... 
    Output Label file: point_1.x point_2.x point_3.x ... point_1.y point_2.y point_3.y ... 
    
## Test Example (WFLW datasets)

![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/0.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/1.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/2.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/3.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/4.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/5.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/6.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/7.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/8.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/9.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/10.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/11.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/12.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/13.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/14.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/15.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/16.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/17.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/18.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/19.jpg)
![image](https://github.com/nianfudong/GCS/blob/master/pytorch/assets/20.jpg)
