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
    <img src="https://github.com/nianfudong/GCS/tree/master/pytorch/assets/1.jpg" >
    <img src="https://github.com/nianfudong/GCS/tree/master/pytorch/assets/2.jpg" >
 
  
    

