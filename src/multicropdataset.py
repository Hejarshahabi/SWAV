# Custom dataloader fro Drill Core dataset
# Author: Laura Elena Cue La Rosa and Hejar Shahabi

import random
from logging import getLogger

import numpy as np
from torch.utils.data import Dataset
import torch
import kornia.augmentation as K
import torch.nn as nn

logger = getLogger()



class DatasetNoCrop(Dataset):
    def __init__(self, 
                 img_path, 
                 mask_path,
                 bands,
                 Num_Bands, 
                 samples=None,
                 labels=None,
                 lab_ind = False):

        super(DatasetNoCrop, self).__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.bands=bands
        self.Num_bands=Num_bands
        self.samples = samples
        self.lab_ind = lab_ind
        
        
    def __len__(self):
        if self.samples:
            return self.samples
        else:
            return len(self.img_path)
        
    def __getitem__(self, idx):
        patch_img = np.load(self.img_path[idx])
        patch_img= patch_img[:,:,self.bands] 
        
        patch_img = torch.from_numpy(patch_img.astype(np.float32)).permute(2,0,1).contiguous()
        
        if self.lab_ind:
            mask_img = np.load(self.mask_path[idx])
            mask_img = torch.from_numpy(mask_img.astype(np.int64))
            return patch_img, mask_img
        
        return patch_img
    
    
class MultiCrop(Dataset):
    def __init__(self, 
                 img_path,
                 mask_path, 
                 size_crops,
                 nmb_crops,
                 bands
                 Num_bands
                 min_scale_crops,
                 max_scale_crops,
                 size_dataset=-1,
                 pil_blur=False,
                 samples=None,
                 labels=None):

        super(MultiCrop, self).__init__()
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.img_path = img_path
        self.mask_path = mask_path
        self.bands=bands
        self.samples = samples
        
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = K.RandomResizedCrop(size=(size_crops[i],size_crops[i]), 
                                                    scale=(min_scale_crops[i], max_scale_crops[i]), 
                                                    same_on_batch=False)


            trans.extend([nn.Sequential(randomresizedcrop,
                                        K.RandomHorizontalFlip(p=0.5,same_on_batch=False),
                                        K.RandomVerticalFlip(p=0.5,same_on_batch=False),
                                        K.RandomRotation(degrees=(90.0,90.0)),
                                        K.RandomRotation(degrees=(180.0,180.0)),
                                        K.RandomRotation(degrees=(270.0,270.0)),
                                        K.RandomBoxBlur(kernel_size=(3, 3),p=0.5),
                                        K.RandomGaussianBlur((3, 3), (0.1, 2.0),p=0.5),
                                        K.RandomElasticTransform(kernel_size=(5, 5), sigma=(32.0, 32.0), alpha=(1.0, 1.0),p=0.5),
                                        K.RandomPerspective(0.5, p=0.5),
                                        K.RandomThinPlateSpline(p=0.5))]*nmb_crops[i])        
        self.trans = trans

        
    def __len__(self):
        if self.samples:
            return self.samples
        else:
            return len(self.img_path)
        
    def __getitem__(self, idx):
        patch_img = np.load(self.img_path[idx])
        patch_img = patch_img[:,:,self.bands]
        patch_img = np.moveaxis(patch_img,2,0)
        patch_img = torch.from_numpy(patch_img.astype(np.float32))
        multi_crops = list(map(lambda trans: trans(patch_img), self.trans))
        multi_crops = [torch.reshape(img,(img.size()[1],img.size()[2],img.size()[3])) for img in multi_crops]
        
        return multi_crops
        
            
