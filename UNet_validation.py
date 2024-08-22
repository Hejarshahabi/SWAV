i################### Custom UNet based on SWAV feature extractor code ############################

##########################################################
# Developed  by: Hejar Shahabi
##########################################################
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mport torch 
import numpy as np
import os 
import glob
import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import transforms
from UNet_train import UNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Implementation of custom Unet")
#########################
#### data parameters ####
#########################
parser.add_argument('--img_path', default="/.../", type=str,
                        help="Path containing the images npy format")
parser.add_argument('--mask_path', default="/.../", type=str,
                    help="Path containing the masks npy format")
parser.add_argument('--bands', default=14, type=int,
                    help="number of bands")
parser.add_argument('--Mean', default=[], type=list,
                    help="mean value of all dataset")
parser.add_argument('--SD', default=[], type=list,
                    help="SD value of all dataset")
parser.add_argument("--batch_size", default=256, type=int,
                    help="batch size")
parser.add_argument('--saved_model_dict', default="/.../*.pth", type=str,
                    help="trained_model_dict")
parser.add_argument('--feature_extractor', default="/.../*.pth", type=str,            
                    help="feature extractor directory")

global args
args = parser.parse_args("")
class patch(Dataset):
      def __init__(self, img , mask, mean, SD):
    self.img=img
    self.mask=mask
    self.mean=mean
    self.SD=SD
  def __len__(self):
    return (len(os.listdir(self.img)))
  
  def __getitem__(self, index):

    lis_img=sorted(glob.glob(self.img+'*.npy')) 
    lis_mask= sorted(glob.glob(self.mask+'*.npy'))#os.listdir(self.mask)

    img_patch=np.load(lis_img[index])
    img_patch=np.transpose(img_patch,(2,0,1))
    img_patch=torch.tensor(img_patch)
    transform=transforms.Normalize(self.mean, self.SD)
    img_patch=transform(img_patch)

    mask_patch=torch.tensor(np.load(lis_mask[index]))
    mask_patch=torch.nn.functional.one_hot(mask_patch.to(torch.int64),2)
    mask_patch=mask_patch.permute(2,0,1)


    return img_patch,mask_patch



#validating the trained model

def main():
    global args
    args = parser.parse_args("")
    model=UNet(2).to(device)
    dic=torch.load(args.saved_model_dict)
    model.load_state_dict(dic["model_state_dicts"])
    los=nn.BCELoss()
    batch =patch(args.img_path, args.mask_path, args.Mean, args.SD)
    val_loader=DataLoader(dataset=batch, batch_size=args.batch_size, shuffle=True)
    model.eval()
    running_loss=0
    correct=0
    total=0
    with torch.no_grade():
      for data in tqdm.tqdm(val_loader):
            patch,label=data[0].to(device),data[1].to(device)
            outputs=model(patch)
            loss=los(outputs, label)
            running_loss+=loss.item()
            -, predicted= outputs.max(1)
            total+=label.size(0)
            correct+=predicted.eq(label).sum().item()
    test_loss=running_loss/len(testloader)
    accu=100.*correct/total
    eval_losses.append(test_loss)
    eval_accu.append(accu)

  print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu)) 
   


if __name__ == "__main__":
       main()

