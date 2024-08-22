################### Custom UNet based on SWAV feature extractor code ############################

##########################################################
# Developed  by: Hejar Shahabi
##########################################################
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.






import torch 
import numpy as np
import os 
import glob
import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import transforms
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
parser.add_argument('--feature_extractor', default="/.../*.pth", type=str,
                    help="feature extractor directory")
#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=500, type=int, 
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=256, type=int,
                    help="batch size")
parser.add_argument("--base_lr", default=0.05, type=float, help="base learning rate")
parser.add_argument("--patience", default=5, type=int, help="patience treshold for learning rate")

#########################
#### other parameters ###
#########################
parser.add_argument("--best_model", default='/.../best_model.pth', type=str,
                    help="save directory for the best model")
parser.add_argument("--best_dict", type=str, default='/.../best_dict.pth',
                    help="save directory for the best model dict")

#custom data loader to load image patches with numpy format
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


# Custom UNet model based on SWAV feature extractor. In this model, the encoder part
# which is a ResNet-18 model trained in SWAV algorithm, is used and its weights/features are freezed.
#  However, the decoder part is developed to be a light decoder.
# in this model only decoder part is trained. 

class UNet(nn.Module):
    def __init__(self, n_class=2):
        super().__init__()
        global args
        args = parser.parse_args("")
        state_dict=args.feature_extractor
        state_dict=torch.load(state_dict,map_location=torch.device('cuda'))
        encoder=models.resnet18(pretrained=False).cuda()
        encoder.conv1=nn.Conv2d(args.bands,64,kernel_size=(3,3),padding=(1,1), bias=False)
        encoder.load_state_dict(state_dict,strict=False)

        self.base_model=encoder.cuda()
        for param in self.base_model.parameters():
          param.requires_grad=False
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]
        
        self.up1=nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(256,256,3,padding=1),
                               nn.BatchNorm2d(256),
                               nn.ReLU(),
                               nn.Conv2d(256,256,3,padding=1),
                               nn.BatchNorm2d(256),
                               nn.ReLU(),
                               nn.Conv2d(256,256,3,padding=1),
                               nn.BatchNorm2d(256),
                               nn.ReLU(),
                               nn.Conv2d(256,256,3,padding=1),
                               nn.BatchNorm2d(256),
                               nn.ReLU())
        self.up2=nn.Sequential(nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(128,128,3,padding=1),
                               nn.BatchNorm2d(128),
                               nn.ReLU(),
                               nn.Conv2d(128,128,3,padding=1),
                               nn.BatchNorm2d(128),
                               nn.ReLU(),
                               nn.Conv2d(128,128,3,padding=1),
                               nn.BatchNorm2d(128),
                               nn.ReLU(),
                               nn.Conv2d(128,128,3,padding=1),
                               nn.BatchNorm2d(128),
                               nn.ReLU())
        self.up3=nn.Sequential(nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(64,64,3,padding=1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.Conv2d(64,64,3,padding=1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.Conv2d(64,64,3,padding=1),
                               nn.BatchNorm2d(64),
                               nn.ReLU())
        self.up4=nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(64,64,3,padding=1),
                               nn.BatchNorm2d(64),
                               nn.ReLU(),
                               nn.Conv2d(64,32,3,padding=1),
                               nn.BatchNorm2d(32),
                               nn.ReLU())
        self.last=nn.Sequential(
                               nn.Conv2d(96,32,3,padding=1),
                               nn.BatchNorm2d(32),
                               nn.ReLU(),
                               nn.Conv2d(32,16,3,padding=1),
                               nn.BatchNorm2d(16),
                               nn.ReLU())
        
        self.conv_last = nn.Conv2d(16, n_class, 1,1)

    def forward(self, input):
        #print(input.shape)
        x1=self.layer0(input)
        #print("x1", x1.shape)
        x2=self.layer1(x1)
        #print("x2",x2.shape)
        x3=self.layer2(x2)
        #print("x3",x3.shape)
        x4=self.layer3(x3)
        #print("x4",x4.shape)
        x5=self.layer4(x4)
        #print("x5",x5.shape)
        #upsampling
        up4= self.up1(x5)
        #print(up4.shape)
        up4_con=torch.cat([up4,x4],dim=1)
        #print(up4_con.shape)
        up5=self.up2(up4_con)
        #print(up5.shape)
        up3_con=torch.cat([up5,x3],dim=1)
        #print(up3_con.shape)
        up6= self.up3(up3_con)
        #print(up6.shape)
        up2_con=torch.cat([up6,x2],dim=1)
        up7= self.up4(up2_con)
        up1_con=torch.cat([up7,x1],dim=1)
        last=self.last(up1_con)
        last_sig=torch.sigmoid(self.conv_last(last))

        return last_sig

#training the model for image segmentation  
def main():
    global args
    args = parser.parse_args("")
    model=UNet(2).to(device)
    los=nn.BCELoss()
    ls=1
    optimize=optim.Adam(model.parameters(),lr=args.base_lr)
    scheduler=optim.lr_sechduler.ReduceLROnPlateau(optimize, patience=args.patience, verbose=True)
    batch =patch(args.img_path, args.mask_path, args.Mean, args.SD)
    test_loader=DataLoader(dataset=batch, batch_size=args.batch_size, shuffle=True)
    model.train()   
    for epoch in tqdm.tqdm(range(args.epochs)):
        losses=[]      
        for ii, (data, target) in enumerate(train_loader):                         
            data, target = data.float().cuda(),target.float().cuda()
            #print(data.shape, target.shape)
            optimize.zero_grad()
            output = model(data)  
            loss = los(output, target)
            loslist.append(loss.item())
            loss.backward()
            optimize.step()
            if loss.item()<ls:
                torch.save(model, args.best_model)
                torch.save(
                    {'epoch': epoch,
                'model_state_dict': un.state_dict(),
                'optimizer_state_dict': optimize.state_dict(),
                'loss': loss.item(),
                }, args.best_dict)
                ls=loss.item()
                losses=append(loss.item())
            mean_loss=sum(losses)/len(losses)  
            scheduler.step(mean_loss)        
        print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))


if __name__ == "__main__":
       main()

