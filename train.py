import torch
from dataloader import Dataset
import torch.nn as nn
import os
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
import math
import time
from torchvision.models import vgg16
from MLDANet import deformable_transformer_CD_net
from tqdm import tqdm

from tensorboardX import SummaryWriter
import dataloader
import numpy as np

batch_size=10
epoch=150
lr_o=1e-4
save_iter=2
set_snapshot_dir="./snapshot/seg/"
set_num_workers=4
set_momentum=0.9
set_weight_decay=0.001
device = torch.device('cuda:0')


def loss_calc(pred,label):
    label=torch.squeeze(label,dim=1)
    pred=torch.squeeze(pred,dim=1)
    loss=nn.BCELoss()
    return loss(pred,label)

def lr_schedule_cosdecay(t,T,init_lr=lr_o):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr


def main():
    writer=SummaryWriter(comment="detrcd on datasetSecond enc_8")
    device = torch.device('cuda:0')

    model=deformable_transformer_CD_net(num_classes=1)
    model.to(device)
    model=nn.DataParallel(model,device_ids=[0,1])

    trainloader=data.DataLoader(
            Dataset(path_root="./datasetlevircd/",mode="train"),
            batch_size=batch_size,shuffle=True,num_workers=set_num_workers,pin_memory=True)

    optimizer=optim.Adam(model.parameters(),lr=lr_o)

    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='min',verbose=True,patience=5,cooldown=3,min_lr=1e-8,factor=0.5)
    for i in range(epoch):
        torch.cuda.empty_cache()
        loss_list=[]
        model.train()
        total_num=0
        true_num=0
        for batch_id,batch in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            sst1,sst2,label=batch
            sst1=sst1.to(device)
            sst2=sst2.to(device)
            label=label.to(device)
            pred=model(sst1,sst2)
            loss=loss_calc(pred,label)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            pred = torch.where(pred > 0.5, one, pred)
            pred = torch.where(pred <= 0.5, zero, pred)
            pred_cm=pred.detach().cpu().squeeze().numpy()

            label_cm=label.detach().cpu().squeeze().numpy()
            pred_cm =pred_cm.astype(np.int64)
            label_cm =label_cm.astype(np.int64)
            true_num=true_num+np.sum(np.array(pred_cm ==label_cm).astype(np.int64))
            total_num = total_num + pred_cm.shape[0]*pred_cm.shape[1]*pred_cm.shape[2]
        
        writer.add_scalar('scalar/OA',true_num/total_num,i)
        scheduler.step(sum(loss_list)/len(loss_list))
        lr=optimizer.param_groups[0]['lr']
        print(f'epoch={i} | loss={sum(loss_list)/len(loss_list):.7f} | lr={lr:.7f}')
        writer.add_scalar('scalar/train_loss',sum(loss_list)/len(loss_list),i)
        

            

if __name__=="__main__":
    main()


    
    
