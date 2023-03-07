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
    
    def get_img_mask(self, json_obj,key):
        
        img = torchvision.io.read_image('data'+"/"+json_obj[key]['filename'])
        c,w,h = img.shape
        # print(c,w,h)
        mask = torch.zeros(size=(1,w,h))
        for key,region in json_obj[key]['regions'].items():
                if region['region_attributes']['body_part']=='body':
                        rr, cc = skimage.draw.polygon(region['shape_attributes']['all_points_y'], region['shape_attributes']['all_points_x'])
                        try:    # some of the masks were right on the edge of the image and thus I needed to catch those cases
                                        mask[0, rr, cc] = 1     # this sets the mask = True for this area
                        except IndexError:
                                        # print(rr,cc)
                                        # # prune rr and cc to 
                                        # rr = [r for r in rr if r<w]
                                        # cc = [c for c in cc if c<h]
                                        # mask[0,rr-1, cc-1] = 1
                                        continue

        return img,mask
    
    def get_mask(self,img,obj):
        mask = torch.zeros((1,img.shape[1],img.shape[2]))

        all_points_y =obj['regions']['0']['shape_attributes']['all_points_y']
        all_points_x =obj['regions']['0']['shape_attributes']['all_points_x']
        rr, cc = skimage.draw.polygon(all_points_y,all_points_x)
        try:    # some of the masks were right on the edge of the image and thus I needed to catch those cases
                    mask[0, rr, cc] = 1     # this sets the mask = True for this area
        except IndexError:
                    mask[0,rr-1, cc-1] = 1
        return mask
    
    def __getitem__(self, index) :
        obj = self.json_obj[self.json_keys[index]]
        # get img
        filename = obj['filename']
        img_path = os.path.join(self.img_dir,filename)
        img,mask = self.get_img_mask(self.json_obj,self.json_keys[index])

        #   resize
        re_size = torchvision.transforms.Resize(size = (640,640))
        img = re_size(img)
        img = img/255.
        mask = re_size(mask)

        return img,mask