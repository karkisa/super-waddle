import torch
import torchvision
import pytorch_lightning as pl
import os
import pandas as pd
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import pdb
class classifier(pl.LightningModule):
    def __init__(self, 
                 model,
                 bs,
                 ds,
                 fold,
                 df_path,
                 wandb_run=None,
                 shuffle=False,
                 criterion=smp.losses.JaccardLoss(mode= 'binary'),
                 LR=2e-3
                 ) :
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.ds= ds
        self.bs = bs 
        self.fold = fold
        self.df = pd.read_csv(df_path)
        self.train_imgs_paths = self.df[self.df["kfold"]!=self.fold]["img_path"]
        self.train_mask_paths = self.df[self.df["kfold"]!=self.fold]['mask_path']
        self.val_img_paths = self.df[self.df["kfold"]==self.fold]["img_path"]
        self.val_mask_paths = self.df[self.df["kfold"]==self.fold]["mask_path"]
        self.shuffle =shuffle
        self.wandb_run = wandb_run
        self.LR=LR

    def train_dataloader(self) :
        train_ds = self.ds(self.train_imgs_paths,self.train_mask_paths)
        train_loader = DataLoader(train_ds,
                                  batch_size=self.bs,
                                  num_workers=4,
                                  shuffle=self.shuffle)
        return train_loader
    
    def val_dataloader(self) :
        val_ds = self.ds(self.val_img_paths,self.val_mask_paths)
        val_loader = DataLoader(val_ds,
                                  batch_size=self.bs,
                                  num_workers=4,
                                  shuffle=self.shuffle)
        return val_loader

    def training_step(self, batch, batch_id) :
        imgs , masks =batch
        outputs = self.model(imgs)
        loss = self.criterion(outputs, masks)
        if self.wandb_run:
            self.wandb_run.log({"train": {"loss":loss}},commit=True)
        return loss
    
    def validation_step(self, batch, batch_id) :
        imgs , masks =batch
        outputs = self.model(imgs)
        # pdb.set_trace()
        loss = self.criterion(outputs, masks)
        if self.wandb_run:
            self.wandb_run.log({"val": {"loss":loss}},commit=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=self.LR)
    
