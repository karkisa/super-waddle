import torch
import torchvision
import pytorch_lightning as pl
from torch.utils.data import Dataset
import random,os
import numpy as np
import skimage
import json, pdb

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
                 json_path,
                 json_keys,
                 img_dir
                 ) -> None:
        super().__init__()
        self.json_obj = json.load(open(json_path))
        self.json_keys = json_keys
        self.img_dir = img_dir

    def __len__(self):
        return len(self.json_keys)
    
    def read_img(self,path):
        img = torchvision.io.read_image(path)
        img = img/255.
        return img 
    
    def __getitem__(self, index) :
        obj = self.json_obj[self.json_keys[index]]
        # get img
        filename = obj['filename']
        img_path = os.path.join(self.img_dir,filename)
        img = self.read_img(img_path)

        # get mask
        mask = torch.zeros((1,img.shape[1],img.shape[2]))
        all_points_y =obj['regions']['0']['shape_attributes']['all_points_y']
        all_points_x =obj['regions']['0']['shape_attributes']['all_points_x']
        rr, cc = skimage.draw.polygon(all_points_y,all_points_x)
        try:    # some of the masks were right on the edge of the image and thus I needed to catch those cases
                    mask[0, rr, cc] = 1     # this sets the mask = True for this area
        except IndexError:
                    mask[0,rr-1, cc-1] = 1

        #   resize
        re_size = torchvision.transforms.Resize(size = (640,640))
        img = re_size(img)
        mask = re_size(mask)
        
        return img,mask