import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import os, pdb
import pandas as pd
from sklearn import model_selection

def read_img(path):
    img = torchvision.io.read_image(path)
    img = img/255. 
    return img

def create_df(img_folder,mask_fodler):
    df = pd.DataFrame(columns=["img_path",'mask_path'])    
    
    for img_name in os.listdir(img_folder):
        df = df.append({'img_path': os.path.join(img_folder,img_name),"mask_path": os.path.join(mask_fodler,img_name)}, ignore_index=True)

    return df

def create_folds(data, num_splits):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1
    
    kf = model_selection.KFold(n_splits=num_splits)

    for f, (t_, v_) in enumerate(kf.split(X=data)):
        data.loc[v_, 'kfold'] = f
 
    return data

if __name__=="__main__":
    img_folder = '/Users/sagar/Desktop/Ace/MiniapplesSegmentation/detection/train/images'
    mask_folder = '/Users/sagar/Desktop/Ace/MiniapplesSegmentation/detection/train/masks'
    save_path = '/Users/sagar/Desktop/Ace/MiniapplesSegmentation/detection/train/train.csv'
    df = create_df(img_folder,mask_folder)
    df = create_folds(df,5)
    df.to_csv(save_path)
    pdb.set_trace()

    