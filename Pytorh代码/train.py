from config import Config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import sys
import torch
import numpy as np
from tensorboardX import SummaryWriter
from utils import My_Dataset,get_time_dif
from models import *
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn.functional as F
import torchvision.models as models
from models.UuXL import U_Net

class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # 加载预训练的VGG16网络
        self.vgg = models.vgg16(pretrained=True).features[:23] #创建了一个预训练的VGG16模型，并仅保留其前23层特征提取部分。
        # 冻结VGG16的所有参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        #在后续的训练过程中，VGG16的参数将不会更新，其作为固定的特征提取器。

    def forward(self, x, y):
        # 使用 VGG16 网络提取高层特征
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        # 计算感知损失
        loss = F.mse_loss(x_vgg, y_vgg)   #感知损失通常用于图像生成或图像转换任务，通过比较高层特征的相似性来提高生成结果的质量。
        return loss
    ##这段代码定义了一个感知损失函数PerceptualLoss，它使用预训练的VGG16模型来提取输入图像的高层特征，并计算预测输出和目标之间的均方误差，作为感知损失。
    ##该损失函数通常用于图像生成任务的训练中，以改进生成结果的质量。

def calculate_psnr(x, y):
    """
    计算两个张量的平均 PSNR 值。

    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
        y (torch.Tensor): 目标张量，形状为 (batch_size, channels, height, width)。

    Returns:
        float: 平均 PSNR 值。
    """
    # 将张量转换为 numpy 数组
    x_np = np.transpose(x.cpu().detach().numpy(), (0, 2, 3, 1))
    y_np = np.transpose(y.cpu().detach().numpy(), (0, 2, 3, 1))
    # 计算每个样本的 PSNR
    psnr_list = []
    for i in range(x.shape[0]):

        psnr = compare_psnr(y_np[i], x_np[i], data_range=1.0)
        psnr_list.append(psnr)

    # 计算平均 PSNR
    mean_psnr = np.mean(psnr_list)

    return mean_psnr

def calculate_ssim(x, y):
    """
    计算两个张量的平均 SSIM 值。

    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, channels, height, width)。
        y (torch.Tensor): 目标张量，形状为 (batch_size, channels, height, width)。

    Returns:
        float: 平均 SSIM 值。
    """
    # 将张量转换为 numpy 数组
    x_np = np.transpose(x.cpu().detach().numpy(), (0, 2, 3, 1))
    y_np = np.transpose(y.cpu().detach().numpy(), (0, 2, 3, 1))

    # 计算每个样本的 SSIM
    ssim_list = []
    for i in range(x.shape[0]):
        # print(y_np[i].shape)
        ssim = compare_ssim(y_np[i], x_np[i], multichannel=True, data_range=1.0,channel_axis=2)
        ssim_list.append(ssim)

    # 计算平均 SSIM
    mean_ssim = np.mean(ssim_list)

    return mean_ssim





def train(config, model, train_iter, dev_iter, test_iter):
    writer = SummaryWriter(log_dir=config.log_dir)
    start_time = time.time()
    model.train()

    optimizer =torch.optim.SGD(mynet.parameters(),lr=config.learning_rate,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,2, gamma=0.9, last_epoch=-1)#每2个epoch学习率衰减为原来的一半
    #每2个 epoch 结束时将学习率乘以 0.9，从而使学习率逐渐减小，以更好地控制模型参数的更新。

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')    #正无穷大被视为大于任何有限的数值
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    ce=nn.MSELoss()
    for epoch in range(config.num_epochs):
        loss_list=[]#承接每个batch的loss
        psnr_list=[]
        ssim_list=[]

        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            optimizer.zero_grad()
            #print(labels)
            # 定义感知损失
            perceptual_loss = PerceptualLoss().to(config.device)
            # 计算感知损失
            ploss = perceptual_loss(outputs, labels)
            closs = ce(outputs, labels)
            loss=(ploss+closs)/2

            loss.backward()
            optimizer.step()
            # 计算 PSNR 和 SSIM
            psnr = calculate_psnr(outputs, labels)
            ssim = calculate_ssim(outputs, labels)
            writer.add_scalar('train/loss_iter', loss.item(),total_batch)
            writer.add_scalar('train/psnr_iter', psnr,total_batch)
            writer.add_scalar('train/ssim_iter', ssim,total_batch)
            msg1 = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train PSNR: {2:>5.2},  Train SSIM: {3:>5.2}'#定义一个格式化输出的字符串msg1
            if total_batch%20==0:
                print(msg1.format(total_batch, loss.item(),psnr,ssim))
            loss_list.append(loss.item())
            psnr_list.append(psnr)
            ssim_list.append(ssim)



            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过2000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

        dev_loss,dev_psnr,dev_ssim = evaluate(config, model, dev_iter)#model.eval()
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            improve = '*'
            last_improve = total_batch
        else:
            improve = ''
        time_dif = get_time_dif(start_time)
        epoch_loss=np.mean(loss_list)
        epoch_psnr=np.mean(psnr_list)
        epoch_ssim=np.mean(ssim_list)
        msg2 = 'EPOCH: {0:>6}, Train Loss: {1:>5.2}, Train PSNR: {2:>5.2}, Train SSIM: {3:>5.2} ,Val Loss: {4:>5.2}, Val PSNR: {5:>5.2}, Val SSIM: {6:>5.2}  Time: {7}'
        print(msg2.format(epoch+1,epoch_loss,epoch_psnr,epoch_ssim,dev_loss,dev_psnr,dev_ssim,time_dif))
        writer.add_scalar('train/loss_epoch',epoch_loss, epoch)
        writer.add_scalar('train/psnr_epoch', epoch_psnr, epoch)
        writer.add_scalar('train/ssim_epoch', epoch_ssim, epoch)

        writer.add_scalar('val/loss_epoch', dev_loss, epoch)
        writer.add_scalar('val/psnr_epoch', dev_psnr, epoch)
        writer.add_scalar('val/ssim_epoch', dev_ssim, epoch)

        model.train()
        scheduler.step()
        print('epoch: ', epoch, 'lr: ', scheduler.get_last_lr())

    test(config, model, test_iter)


def test(config, model, test_iter):
    # 测试函数
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_loss ,test_psnr,test_ssim= evaluate(config, model, test_iter, test=True)
    print('***********************************************************')
    msg = 'Test Loss: {0:>5.2},Test PSNR: {1:>5.2},Test SSIM: {2:>5.2}'
    print(msg.format(test_loss,test_psnr,test_ssim))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    psnr_total=0
    ssim_total=0
    ce=nn.MSELoss()
    with torch.no_grad():
        for texts, labels in data_iter:
            #print(texts)

            outputs = model(texts)

            # 定义感知损失
            perceptual_loss = PerceptualLoss().to(config.device)
            # 计算感知损失
            ploss = perceptual_loss(outputs, labels)
            closs = ce(outputs, labels)
            loss = (ploss + closs) / 2
            loss_total += loss

            # 计算 PSNR 和 SSIM
            psnr = calculate_psnr(outputs, labels)
            ssim = calculate_ssim(outputs, labels)
            psnr_total+=psnr
            ssim_total += ssim
            # print(outputs)
            # print(predic)
            # print(labels)
            # print('*************************')
    return loss_total / len(data_iter),psnr_total / len(data_iter),ssim_total / len(data_iter)



if __name__ == '__main__':
    config = Config()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print("Loading data...")


    train_data=My_Dataset(config.traincsv,config,1)
    dev_data = My_Dataset(config.testcsv,config,1)
    test_data = My_Dataset(config.valcsv,config,1)


    train_iter=DataLoader(train_data, batch_size=config.batch_size,shuffle=True)   ##训练迭代器
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size,shuffle=True)      ###验证迭代器
    test_iter = DataLoader(test_data, batch_size=config.batch_size,shuffle=True)   ###测试迭代器
    # 训练mynet =get_transNet(3, config.modelname)
    mynet = U_Net()
    ## 模型放入到GPU中去
    mynet= mynet.to(config.device)
    print(mynet.parameters)

    #训练结束后可以注释掉train函数只跑test评估模型性能
    #test(config, mynet, test_iter)
    train(config, mynet, train_iter, dev_iter, test_iter)
    test(config, mynet, test_iter)

#tensorboard --logdir=log/R50-ViT-B_16 --port=6006
##tensorboard --logdir=log/ViT-B_16 --port=6006

#tensorboard --logdir=r'E:\TransUnet\Transu\log\xinxin' --port=6006
#tensorboard --logdir=E:\TransUnet\Transu\log\xinxin --port=6006



