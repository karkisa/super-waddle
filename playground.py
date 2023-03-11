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
    bs=8
    fold=0
    json_path = 'data/via_region_data.json'
    img_dir = 'data'
    kfold_csv  = 'kfold_keys.csv'
    wandb_run = wandb.init(
                        project='Segmentation',
                        group=str(fold),
                        name='exp4',
                        # resume=True
                    )
    model =  smp.Unet(
                            encoder_name='efficientnet-b3',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                            activation='sigmoid'
                 )
    
    Classifier=classifier(
        ds= base_pipe,
        model=model,
        bs=bs,
        json_path=json_path,
        img_dir=img_dir,
        kfold_keys_csv=kfold_csv,
        fold=fold,
        wandb_run=wandb_run,
    )
    
    Trainer=pl.Trainer(
                        # devices=1,
                        # accelerator="mps",
                        max_epochs=35,
                        log_every_n_steps=10,
                        accumulate_grad_batches=4
                      )
    Trainer.fit(Classifier)


if __name__=="__main__":
    main()