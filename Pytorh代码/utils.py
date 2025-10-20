import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import cv2
from config import Config
# print(torch.__version__)

class My_Dataset(Dataset):
    def __init__(self,path,config,iftrain):#### 读取数据集
        self.config=config
        #启用训练模式，加载数据和标签

        self.iftrain=iftrain
        df = pd.read_csv(path)
        self.img_path = df['img'].to_list() #[img]
        self.mask_path = df['mask'].to_list()  # [img]

        self.transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3,sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Resize((512,512)),     #transunet等
            # transforms.Resize((256, 256)),
            transforms.Normalize((0.485, 0.456, 0.406),  # using ImageNet norms
                                 (0.229, 0.224, 0.225))])


    def __getitem__(self, idx):
        img=Image.open(self.img_path[idx]).convert("RGB")
        # img=self.transform(img)
        img = img.resize((512,512))
        # img = img.resize((256, 256))
        img=np.array(img).transpose((2,0,1))/255.    #将图像数据转换为NumPy数组，并通过 transpose((2,0,1)) 调整通道顺序为(C, H, W)。
        img=torch.tensor(img,dtype=torch.float32)     # 将 Numpy 数组转换为 PyTorch 张量，并指定数据类型为 torch.float32，即单精度浮点型。
        # print(img.shape)
        mask = Image.open(self.mask_path[idx]).convert("RGB")    #标签数据
        mask = mask.resize((512,512))
        # mask = mask.resize((256,256))
        mask=np.array(mask).transpose((2,0,1))/255.
        mask=torch.tensor(mask,dtype=torch.float32)

        return img.to(self.config.device),mask.to(self.config.device)  #返回图像数据和对应的标签数据，并将它们都移动到配置对象config中定义的设备上

    def __len__(self):
        return len(self.img_path)#总数据长度

def get_time_dif(start_time):   #计算时间差（运行时长）

    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__=='__main__':
    config=Config()
    train_data=My_Dataset(config.valcsv,config,1)
    train_iter = DataLoader(train_data, batch_size=1)
    n=0
    for a,b in train_iter:
        n=n+1
        print(n,a.shape,b.shape)
        #print(y)
        print('************')
        break