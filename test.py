# get images
#  for every image 
#       run the model on it

import os
import torch 
import torchvision
from segmentation_models_pytorch import Unet

def main():
    test_folder = ''
    img_paths =[ os.path.join(test_folder,image_name) for image_name in os.listdir(test_folder)]
    model = model =  Unet(
                            encoder_name='efficientnet-b3',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            activation='sigmoid',
                 )
    
    

    return 



if __name__=="__main__":
    main()