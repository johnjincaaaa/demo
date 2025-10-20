from models import *
from config import Config
from torch.utils.data import DataLoader
from utils import My_Dataset,get_time_dif
from models import *
from torchvision import transforms as transforms
from PIL import Image
import json
import torch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from models.UuXL import U_Net, R2U_Net, AttU_Net, R2AttU_Net,U_Net


def calculate_psnr(x, y):
    x_np = np.transpose(x.cpu().detach().numpy(), (0, 2, 3, 1))
    y_np = np.transpose(y.cpu().detach().numpy(), (0, 2, 3, 1))
    psnr = compare_psnr(y_np[0], x_np[0], data_range=1.0)
    return psnr

def calculate_ssim(x, y):
    x_np = np.transpose(x.cpu().detach().numpy(), (0, 2, 3, 1))
    y_np = np.transpose(y.cpu().detach().numpy(), (0, 2, 3, 1))
    ssim = compare_ssim(y_np[0], x_np[0], multichannel=True, data_range=1.0, channel_axis=2)

    return ssim

config=Config()
# img=Image.open('sy/5.png').convert("RGB")#预测图片的路径
img = Image.open('balancedCeshi/picture/u134.png').convert("RGB")   #6,134,184
# img = Image.open('WZCceshi/picture/u43.png').convert("RGB")
# img=Image.open('sy/yuantu2.png').convert("RGB")#预测图片的路径
# img = Image.open('Zceshi/picture/u53.png').convert("RGB")
# ground_truth_img = Image.open('sy/r604.png').convert("RGB")
# mask=Image.open('sy1/r68.png').convert("RGB")#真实真值图的路径
# imgnpp=np.array(img.resize((512,512)))
# imgnp=np.array(img.resize((256,256)))
imgnp=np.array(img)
# masknp=np.array(mask.resize((512,512)))
# ground_truth_np = np.array(ground_truth_img)

# 获取输入图片的大小
input_height, input_width, _ = imgnp.shape

img = img.resize((512, 512))
# img = img.resize((256, 256))
img = np.array(img).transpose((2, 0, 1)) / 255.
img = torch.tensor(img, dtype=torch.float32)
img=img.unsqueeze(0)#因为单张图片 所以需要增加一个batch维度  将单张图片转换为一个大小为(1,C,H,W)的张量


# ground_truth_img = Image.open('sy/z5.png').convert("RGB")
# ground_truth_img = Image.open('Zceshi/label/r53.png').convert("RGB")    #13/53/70/71
ground_truth_img = Image.open('balancedCeshi/label/r134.png').convert("RGB")
# ground_truth_img = Image.open('WZCceshi/label/r43.png').convert("RGB")
# ground_truth_img = Image.open('sy/r3.png').convert("RGB")
ground_truth_img = ground_truth_img.resize((512, 512))
ground_truth_np = np.array(ground_truth_img).transpose((2, 0, 1)) / 255.
ground_truth = torch.tensor(ground_truth_np, dtype=torch.float32)
ground_truth = ground_truth.unsqueeze(0)


img=img.to(config.device)
model = U_Net()

## 模型放入到GPU中去
model = model.to(config.device)
model.load_state_dict(torch.load(config.save_path))#加载训练好的模型
model.eval()
with torch.no_grad():
    outputs = model(img)
    out=outputs.cpu().numpy()
    out=out[0, :, :, :]
    # print(out.shape)
    out=np.transpose(out, (1, 2, 0))

    # 将numpy数组转换为Image对象
    out=(out*255).astype(np.uint8)
    outpic = Image.fromarray(out)

    outpic_resized = outpic.resize((input_width, input_height))
    # 保存Image对象为图片文件
    outpic_resized.save('output_resized.png')
    # outpic.save('output.png')


# 计算PSNR和SSIM
psnr_value = calculate_psnr(outputs, ground_truth)
ssim_value = calculate_ssim(outputs, ground_truth)

print(f'PSNR: {psnr_value:.2f}')
print(f'SSIM: {ssim_value:.4f}')

f=plt.figure(figsize=(15,5),dpi=500)  #生成一个窗体，窗体的规格可以在这里设置
f.add_subplot(1,2,1)
plt.imshow(imgnp)
plt.title('origin pic')  #子图的标题
plt.xticks([]), plt.yticks([])  #去除坐标轴

# f.add_subplot(1,3,2)
# plt.imshow(masknp)
# plt.title('real pic')  #子图的标题
# plt.xticks([]),plt.yticks([])  #去除坐标轴

f.add_subplot(1,2,2)
plt.imshow(outpic_resized)
# plt.imshow(outpic)
plt.title('predict pic')  #子图的标题
plt.xticks([]),plt.yticks([])  #去除坐标轴
plt.savefig('对比图.png')
plt.show()  #显示