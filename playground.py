import torch 
import torchvision
import wandb
import os
import pytorch_lightning as pl
from pipeline import base_pipe, seed_everything
from logic import classifier
from model import get_model
import segmentation_models_pytorch as smp

def main():
    seed=42
    seed_everything(seed)
    bs=4
    fold=0
    df_path = '/Users/sagar/Desktop/Ace/MiniapplesSegmentation/detection/train/train.csv'
    # model=get_model()
    # wandb_run = wandb.init(
    #                     project='Segmentation',
    #                     group=str(fold),
    #                     name='exp0'
    #                 )
    model =  smp.Unet(
                            encoder_name='efficientnet-b0',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            # classes=3,                      # model output channels (number of classes in your dataset))
                 )
    Classifier=classifier(
        ds= base_pipe,
        model=model,
        bs=bs,
        df_path=df_path,
        fold=fold,
        # wandb_run=wandb_run,
    )
    
    Trainer=pl.Trainer(
                        # devices=1,
                        # accelerator="mps",
                        max_epochs=35,
                        log_every_n_steps=10,
                        
                      )
    Trainer.fit(Classifier)



if __name__=="__main__":
    main()