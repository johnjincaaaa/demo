import os.path
import torch

#asl 28 signn24 src26
#通过修改self.bert_name和self.resnet_name来决定模型的具体结构
class Config(object):
    def __init__(self):

        # self.data = 'dataset/'
        # self.data = '/home/ps/lq/MultiscaleNeth/Bingdatasets/'
        self.data = '/home/ps/lq/MultiscaleNeth/balancedDataset/'
        # self.data = '/home/ps/lq/MultiscaleNeth/WZCdataset/'
        # self.data = 'PreDataset/'  # signn src
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        # 使用默认 GPU（如果有的话）
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 获取所有可用的 GPU 设备

        available_gpus = list(range(torch.cuda.device_count()))
        # 选择要使用的 GPU 设备，这里选择第二个 GPU（如果有的话）
        chosen_gpu = 1  # 修改为你想要使用的 GPU 的索引

        # 确保选择的 GPU 在可用设备列表中
        if chosen_gpu in available_gpus:
            self.device = torch.device(f'cuda:{chosen_gpu}')
        else:
            print(f"Warning: Chosen GPU {chosen_gpu} is not available. Falling back to the default GPU or CPU.")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.device = os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"   # 设备
        # self.device = os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设备
        self.dropout = 0.3     # 随机失活
        self.require_improvement = 1000  # 若超过2000batch效果还没提升，则提前结束训练  ，TransUnet需要设置大一点如10000
        self.num_epochs = 80  # epoch数
        self.batch_size = 16
        # self.learning_rate = 1e-3 #resnet的学习率,最好比bert的学习率略高
        self.learning_rate = 0.05

        self.modelname = 'BalancedDatasets_Unet_lossnet24_80epoch_batchsize8_lr005'  ## closs+0.1*loss2

        if not os.path.exists('save_model'): #检查当前目录下是否存在名为'save_model'的文件夹。如果该文件夹不存在，它将创建一个新的'save_model'文件夹。
            os.makedirs('save_model')

        self.traincsv=self.data+'train.csv' #通过拼接两部分字符串得到完整的训练集文件路径'dataset/train.csv'
        self.testcsv=self.data+'test.csv'
        self.valcsv=self.data+'val.csv'
        self.index_label=self.data+'index.json'

        self.save_path = 'save_model/'+self.modelname+'.ckpt'#保存模型的路径
        self.log_dir= './log/'+self.modelname#tensorboard日志的路径



