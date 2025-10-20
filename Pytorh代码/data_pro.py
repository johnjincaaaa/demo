import os
from sklearn.model_selection import train_test_split
import pandas as pd

'''这是第一步要运行的代码，生成训练测试验证数据集'''
# data='/home/ps/lq/TransuUnet2/Zdataset/'#数据路径
data='/home/ps/lq/TransuUnet2/PreDataset/'#数据路径
testdata='PreCeshi/'
# testdata='ceshi/'
dirs=os.listdir(data)
dirs=[data+x for x in dirs]#所有类别数据路径
# dirs=['dataset/lung']   #自己改的肺部数据集
pic_list=[]#承接所有的
mask_list=[]
for dir in dirs:
    pics=os.listdir(dir+'/picture/')
    #pics=[dir+'/picture/'+x for x in pics]#原始图片的路径
    masks=os.listdir(dir+'/label/')
    res=min(len(masks),len(pics))
    for x in range(1,res+1):
        p=dir+'/picture/'+'u{}.png'.format(x)#原始图片
        m=dir+'/label/'+'r{}.png'.format(x)#标签真值
        if os.path.exists(m) and os.path.exists(p) :
            pic_list.append(p)
            mask_list.append(m)

df=pd.DataFrame()      #使用Python的pandas库来创建一个DataFrame（数据表格）
df['img']=pic_list
df['mask']=mask_list

# train,val=train_test_split(df,test_size=0.1,random_state=0)  #将数据集中的10%作为验证集，剩余的90%作训练集。
train,val=train_test_split(df,test_size=0.11,random_state=0)  #将数据集中的10%作为验证集，剩余的90%作训练集。
# 如果不设置random_state，每次运行代码将得到不同的训练集和验证集

testpic=os.listdir(testdata+'picture/')
testmask=os.listdir(testdata+'label/')
testres=min(len(testmask),len(testpic))
testm=[]
testpic=[]
for x in range(1, testres + 1):
    p = testdata + 'picture/' + 'u{}.png'.format(x)  # 原始图片
    m = testdata + 'label/' + 'r{}.png'.format(x)  # 标签真值
    testm.append(m)
    testpic.append(p)

test=pd.DataFrame()
test['img']=testpic
test['mask']=testm
train.to_csv(data+'train.csv',index=None)   #to_csv()是DataFrame对象的方法，可将DataFrame中的数据保存为CSV文件
val.to_csv(data+'val.csv',index=None)
test.to_csv(data+'test.csv',index=None)
print('训练集数量：',len(train))
print('测试集数量:',len(test))
print('验证集数量:',len(val))
