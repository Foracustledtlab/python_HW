#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import os
import  matplotlib.pyplot  as plt
import zipfile
import numpy as np
import math
import cv2 as cv
get_ipython().system(' pip install scikit-image -i https://mirror.baidu.com/pypi/simple')
from skimage.feature import greycomatrix,greycoprops
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import matplotlib
matplotlib.use('Agg') 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
get_ipython().system(' pip install paddlex -i https://mirror.baidu.com/pypi/simple')
import paddlex as pdx


# In[ ]:


#解压必备文件的函数
def uzip(zipname,unzippath):
    zip_file = zipfile.ZipFile(zipname)
    for names in zip_file.namelist():
        zip_file.extract(names,unzippath)
    zip_file.close()
def fzip(fname,zipname):
    zip_file = zipfile.ZipFile(zipname, 'w' ,zipfile.ZIP_DEFLATED) 
    zip_file.write(fname)
    zip_file.close()
#import sys  
#sys.path.append('/home/aistudio/external-libraries')
#安装新的库固定在文件夹里


# In[ ]:


#模型训练部分 可以跳过不启动这块
from paddlex.det import transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32)
])
eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32),
])
train_dataset = pdx.datasets.VOCDetection(
    data_dir='VOC_add',
    file_list='VOC_add/train_list.txt',
    label_list='VOC_add/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='VOC_add',
    file_list='VOC_add/val_list.txt',
    label_list='VOC_add/labels.txt',
    transforms=eval_transforms)
# 如果要通过VisualDL查看日志页面，下没按这行代码需要执行
# aistudio上需要将日志输出到/home/aistudio/log目录下才可以查看VisuaDL界面
get_ipython().system(' rm -rf ~/log & rm -rf output/faster_rcnn_r50_fpn')
get_ipython().system(' mkdir -p output/faster_rcnn_r50_fpn/vdl_log')
get_ipython().system(' ln -s output/faster_rcnn_r50_fpn/vdl_log ~/log')
import paddle.fluid as fluid
import paddle
num_classes = len(train_dataset.labels) + 1
model = pdx.det.FasterRCNN(num_classes=num_classes,with_fpn=True,backbone='ResNet101_vd'
aspect_ratios=[0.6,1.0,1.6],anchor_sizes=[12,16,20,24,32])
model.train(
    num_epochs=30,train_dataset=train_dataset,
    train_batch_size=2,eval_dataset=eval_dataset,
    learning_rate=0.0025,lr_decay_epochs=[8,16,24,30],
    lr_decay_gamma=0.1,save_interval_epochs=1,metric='VOC',
    save_dir='output/faster_ResNet101',early_stop=True,early_stop_patience=5,pretrain_weights='output/faster_ResNet101/epoch_16',
    use_vdl=True)
metrics, evaluate_details = model.evaluate(eval_dataset=eval_dataset,batch_size=1, epoch_id=None,
metric='VOC',return_details=True)
gt = evaluate_details['gt']
bbox = evaluate_details['bbox']
pdx.det.draw_pr_curve(gt=gt, pred_bbox=bbox)
total_image=open(r'VOCdata/val_list.txt').readlines()
for i in range(len(total_image)):
    image_name=total_image[i].split()[0]
    result = model.predict('VOCdata/'+image_name)
    pdx.det.visualize('VOCdata/'+image_name, result, threshold=0.5, save_dir='image/')


# In[ ]:


#import os
#imageset=os.listdir(path)
#函数部分
def get_picture(imageset):    
    '''产生梯度图与曲度图和灰度图
    imageset:为需要产生这三类图的每张原图路径列表 如：['VOCdata/JPEGImages/001313.jpg']
    '''
    sobel,laplacian,origin=[],[],[]
    for IM in imageset:
        img=cv.imread(IM,0)                    
        img=cv.GaussianBlur(img,(3,3),0)            # CV_8U - 8位无符号整数（0..255）
        sobel.append(cv.Sobel(img,cv.CV_8U,1,1))      
        laplacian.append(cv.Laplacian(img,cv.CV_8U,ksize=3))
        origin.append(img)
    total=[origin,sobel,laplacian]
    return total
def compute(img):    #定义灰度共生矩阵，求特征值
        p=greycomatrix(img, [2], [0,np.pi/8,np.pi/4,np.pi*3/8,np.pi/2,np.pi*5/8,np.pi*3/4,np.pi*7/8])
        Con=greycoprops(p,prop='contrast')
        Diss=greycoprops(p,prop='dissimilarity')
        homo=greycoprops(p,'homogeneity')
        ASM=greycoprops(p,'ASM')
        Ene=greycoprops(p,'energy')
        Corr=greycoprops(p,'correlation')
        total=np.concatenate((Con,Diss,homo,ASM,Ene,Corr))
        means=np.mean(total,axis=1)
        max_min=np.max(total,axis=1)-np.min(total,axis=1)
        outcome=np.concatenate((means,max_min))
        return outcome
def  get_feature_list(imageset,total_picture):
    '''得到最终用于预分类器的图片特征
    total_pircute:包含所有原图的三类变化后的图，即上面经过get_picture后的列表
    ''' 
    feature_list=np.zeros((len(imageset),36))
    origin,sobel,laplacian=total_picture
    for i in range(len(imageset)):
        origin_feartures=compute(origin[i])
        sobel_features=compute(sobel[i])
        laplacian_feartures=compute(laplacian[i])
        feature_list[i]=np.concatenate((sobel_features,origin_feartures,laplacian_feartures))
    return feature_list


# In[ ]:


#分类
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
def get_Proba(train_X,train_y,feature_list):
    '''用于得到预分类后每一站图对应有无息肉类别的概率情况
    train_X,train_y:用于预分类器训练的数据
    feature_list:值用于预测的特征矩阵
    '''
    test_X=feature_list                #test_X=np.load("目标灰度特征.npy")
    model=ExtraTreesClassifier(n_estimators=250)
    model.fit(train_X,train_y)
    Proba= model.predict_proba(test_X)
    #np.save("灰度概率.npy",Proba)
    return Proba
def get_threshold_box(imageset,model,Proba,watershed=0.4):
    '''得到图片的预测框，并与预分类求出的概率相乘，过滤低于分水岭的预测框数据
    watershed:低于该数值的预测框会被过滤
    model:用于预测的faster-rcnn模型
    Proba:预分类得到的对每张图的两类概率
    '''
    prediction_box=[]           
    for IM in imageset:
        result = model.predict(IM)
        prediction_box.append(result)
    outcome=[]
    for i in range(len(prediction_box)):
        ip=[]
        for a in prediction_box[i]:       
            a["score"]*=Proba[i,1]
            if a["score"]> watershed:               #得分分水岭值
                ip.append(a['bbox'])
        outcome.append(ip) 
    return outcome


# In[ ]:


#获取真实框与求出每一个预测框与真实框的IOU
def get_ground_truth(imageset,labels_train_dir):
    '''得到每一张图的真实框的xywh，如果改图无息肉，则真实框为一个空列表
    imageset:为需要产生这三类图的每张原图路径列表 如：['VOCdata/JPEGImages/001313.jpg']
    labels_train_dir:比赛给的857个txt文件的路径目录名 如：'labels_train/IM_0001.txt'则目录名为labels_train'
    '''
    image_path=[]
    for i in range(len(imageset)):
        num=int(imageset[i][19:25])
        image_path.append(os.path.join(labels_train_dir,'IM_'+str(num).zfill(4)+'.txt')) #将每张图路径改成labels_train下的路径
    ground_truth=[]
    for IM in image_path:
        ip=[]
        try:
            fp=open(IM)
            line=list(map(float,fp.read().split()))
            if len(line)==5:
                    ip.append([line[1]*512,line[2]*512,line[3]*512,line[4]*512])
            else:
                    ip.append([line[1]*512,line[2]*512,line[3]*512,line[4]*512])
                    ip.append([line[6]*512,line[7]*512,line[8]*512,line[9]*512])
            ground_truth.append(ip)
        except FileNotFoundError  : 
            ground_truth.append(ip)
    return ground_truth
def classification_compute(ground_truth,prediction_box):
    '''分类准确率的衡量，有预测框视为分类为有息肉类，无预测框视为分类为无息肉类
    ground_truth:指的是真实框的列表
    prediction_truth:指的是预测框的列表
    '''
    TP,FP,FN,TN=0,0,0,0
    prediction_y,true_y=[],[]
    for box in ground_truth:                    #得到真实和预测的图像类别
        if len(box)>0:
            true_y.append(1)
        else :true_y.append(0)
    for box in prediction_box:
        if len(box)>0:
            prediction_y.append(1)
        else :prediction_y.append(0)
    for i in range(len(prediction_y)):
        if prediction_y[i]==1 and true_y[i]==1: TP+=1
        elif prediction_y[i]==1 and true_y[i]==0:FP+=1
        elif prediction_y[i]==0 and true_y[i]==1:FN+=1
        elif prediction_y[i]==0 and true_y[i]==0:TN+=1  
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    accuracy=(TP+TN)/(TP+FP+FN+TN)
    output_dict={'TP':TP,'FP':FP,'FN':FN,'TN':TN,'precision':precision,'recall':recall,'accuracy':accuracy}
    return output_dict 


# In[ ]:


def get_single_IOU(box1,box2):
    '''计算一对框的IOU，box1与box2为预测框与真实框,框内的数据为xywh,格式为[200,200,12,13]
    '''
    x1min, y1min = box1[0] - box1[2]/2.0, box1[1] - box1[3]/2.0
    x1max, y1max = box1[0] + box1[2]/2.0, box1[1] + box1[3]/2.0
    s1 = box1[2] * box1[3]
    x2min, y2min = box2[0] - box2[2]/2.0, box2[1] - box2[3]/2.0
    x2max, y2max = box2[0] + box2[2]/2.0, box2[1] + box2[3]/2.0
    s2 = box2[2] * box2[3]
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xmin,ymin = np.max([x1min, x2min]),np.max([y1min, y2min])
    xmax,ymax = np.min([x1max, x2max]),np.min([y1max, y2max])
    inter_h ,inter_w= np.max([ymax - ymin, 0.]),np.max([xmax - xmin, 0.])
    intersection = inter_h * inter_w
    union = s1 + s2 - intersection
    iou = intersection / union
    return iou


# In[ ]:


#uzip('labels_train.zip','')
#uzip('/home/aistudio/data/data42532/model99.zip','')      #用于解压labels_train和模型
total_image=open(r'VOCdata/val_list.txt').readlines()
for i in range(len(total_image)):
    total_image[i]='VOCdata/'+total_image[i].split()[0]          #将VOC的txt文件转化为文件夹中的图片路径
total_picture=get_picture(total_image)
features=get_feature_list(total_image,total_picture)  #得到测试集特征并用于预分类器分类


# In[ ]:


train_image=open(r'VOCdata/train_list.txt').readlines()
for i in range(len(train_image)):
    train_image[i]='VOCdata/'+train_image[i].split()[0]   #得到预分类器用于训练的特征矩阵和目标向量(测试集1050张图片)
train_picture=get_picture(train_image)
train_X=get_feature_list(train_image,train_picture)
train_y=[]
for IM in train_image:
    num=int(IM[19:25])
    if num >856:
        train_y.append(0)
    else: 
        train_y.append(1)
Proba=get_Proba(train_X,train_y,features)            #得到测试集预训练对每张图的分类概率
model_dir='output/faster_ResNet101/epoch_30'
model = pdx.load_model(model_dir)
bounding_box=get_threshold_box(total_image,model,Proba)
ground_truth=get_ground_truth(total_image,'labels_train')        #得到测试集真实框的xywh


# In[ ]:


imageset=total_image[0:10]
prediction_box=[]           
for IM in imageset:
    result = model.predict(IM)
    prediction_box.append(result)
outcome=[]
for i in range(len(prediction_box)):
    ip=[]
    for a in prediction_box[i]:       
        a["score"]*=Proba[i,1]
        if a["score"]> 0.2:               #得分分水岭值
            ip.append(a['bbox'])
    outcome.append(ip) 


# In[ ]:


{'TP': 5,
 'FP': 0,
 'FN': 0,
 'TN': 5,
 'precision': 1.0,
 'recall': 1.0,
 'accuracy': 1.0}


# In[ ]:


outcome=[]
for i in range(len(prediction_box)):
    ip=[]
    for a in prediction_box[i]:       
        a["score"]*=Proba[i,1]
        if a["score"]> 0:               #得分分水岭值
            ip.append(a)
    outcome.append(ip)


# In[ ]:


import json

with open('outcome.txt','w') as web:
    web.write(json.dumps(metrics)+'\n')
    total_image=open(r'VOCdata/val_list.txt').readlines()
    outcome=[]
    for i in range(len(total_image)):
        image_name=total_image[i].split()[0]
        result = model1.predict('VOCdata/'+image_name)
        outcome.append(result)
    web.write(json.dumps(outcome))


# In[ ]:


total_image=open(r'VOCdata/val_list.txt').readlines()
image_name=total_image[20].split()[0]
result = model1.predict('VOCdata/'+image_name)


# In[ ]:


json.dumps(result)

