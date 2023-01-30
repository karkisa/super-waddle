import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import Dataset
import random,os
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class base_pipe(Dataset):
    def __init__(self,
                 img_paths,
                 mask_paths
                 ) -> None:
        super().__init__()
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)
    
    def read_img(self,path):
        img = torchvision.io.read_image(path)
        img = torchvision.transforms.Resize(size = (1280, 736))(img)
        img = img/255.
        return img 
    
    def __getitem__(self, index) :
        img = self.read_img(self.img_paths[index])
        mask = self.read_img(self.mask_paths[index])
        return img,mask