import torch
import torchvision
import pytorch_lightning as pl
import os
import pandas as pd
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import pdb, wandb
from torchvision.transforms import ToPILImage
import numpy as np
class classifier(pl.LightningModule):
    def __init__(self, 
                 model,
                 bs,
                 ds,
                 fold,
                 img_dir,
                 json_path,
                 kfold_keys_csv,
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
        self.json_path = json_path
        self.img_dir = img_dir
        self.kfold_keys_df = pd.read_csv(kfold_keys_csv)
        self.train_keys = self.kfold_keys_df[self.kfold_keys_df["fold"]!=self.fold]["keys"].values
        self.val_keys = self.kfold_keys_df[self.kfold_keys_df["fold"]==self.fold]["keys"].values
        self.shuffle =shuffle
        self.wandb_run = wandb_run
        self.LR=LR

    def wandb_log_masks(self,original_images,class_labels,prediction_masks,ground_truth_masks,thresh = 0.5):
        
            # pdb.set_trace()
            dict_log = []
            for i,(original_image,prediction_mask,ground_truth_mask) in enumerate(zip (original_images,prediction_masks,ground_truth_masks)):

                dict_log.append(
                                    wandb.Image(original_image.cpu().permute(1,2,0).numpy(), masks=
                                                                            {
                                                                            "predictions" : {
                                                                                "mask_data" : (prediction_mask[0].cpu().detach().numpy() > thresh).astype(float),
                                                                                "class_labels" : class_labels
                                                                            },
                                                                            "ground_truth" : {
                                                                                "mask_data" : ground_truth_mask[0].cpu().detach().numpy(),
                                                                                "class_labels" : class_labels
                                                                            }
                                                                            }
                                                                        )

                                )
            
            wandb.log({"Outputs" : dict_log})
            return
        
    def get_metrices(self,outputs,targets):
        tp, fp, fn, tn = smp.metrics.get_stats(outputs, targets.type(torch.int64), mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        # recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        return iou_score,f1_score
        

    def train_dataloader(self) :
        train_ds = self.ds(self.json_path,self.train_keys,self.img_dir)
        train_loader = DataLoader(train_ds,
                                  batch_size=self.bs,
                                  num_workers=4,
                                  shuffle=self.shuffle)
        return train_loader
    
    def val_dataloader(self) :
        val_ds = self.ds(self.json_path,self.val_keys,self.img_dir)
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
        loss = self.criterion(outputs, masks)
        iou_score,f1_score = self.get_metrices(outputs,masks)
        if self.wandb_run:
            self.wandb_log_masks(imgs,{0: "body"},outputs,masks)
            self.wandb_run.log(
                                {
                                    "val": {
                                                "loss":loss,
                                                "IOU" : iou_score,
                                                "f1" : f1_score
                                            },
                                },
                                commit=True,
                               )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=self.LR)
    
