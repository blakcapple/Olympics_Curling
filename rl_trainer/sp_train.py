"""
监督学习训练
"""
# 添加外部路径
import sys
from pathlib import Path
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)

from os import path 
import os 
import torch 
import torch.nn as nn 
from rl_trainer.algo.network import mlp
import pickle
import numpy as np 
from copy import deepcopy
import wandb
import math
import pdb

def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))

def log_train(train_info, train_step):
    for k, v in train_info.items():
        wandb.log({k:v}, step=train_step)

def train(model, train_data, test_data, epoch, batch_size, device, lr, test_interval=10):
    wandb.init(project="JIDI_Competition", entity="the-one")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_data = torch.tensor(train_data, dtype=torch.float32, device=device)
    test_data =  torch.tensor(test_data, dtype=torch.float32, device=device)
    model.to(device)
    data_num = train_data.shape[0]
    optim_iter_num = int(math.ceil(data_num / batch_size))
    min_test_error = np.inf
    for i in range(epoch):
        perm = np.arange(data_num)
        np.random.shuffle(perm)
        train_data = train_data[perm].clone()
        for j in range(optim_iter_num):

            index = slice(j*optim_iter_num, min((j+1)*optim_iter_num, data_num))
            input = train_data[index, 0] 
            truth = train_data[index, 1]
            predict = model(input)
            train_loss = loss_fn(predict, truth)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            wandb.log({'train_loss':train_loss.item()})
            print(f'Epoch: {i}, Optim_iter: {j}, Train_Loss:{train_loss.item()}')

        if i % test_interval == 0:
            input = test_data[:,0]
            truth = test_data[:,1]
            with torch.no_grad():
                predict = model(input)
                test_error = loss_fn(truth, predict)
                if min_test_error > test_error:
                    min_test_error = test_error
                    save_dir = path.abspath(path.join(path.dirname(path.abspath(__file__)), '../sp_model'))
                    save_pth = os.path.join(save_dir, 'model_best')
                    model.save_model(save_pth)
            wandb.log({'test_error':test_error.item()})
            print(f'Epoch: {i}, test_error:{test_error.item()}')
        
        if i % 100 == 0:
            save_dir = path.abspath(path.join(path.dirname(path.abspath(__file__)), '../sp_model'))
            save_pth = os.path.join(save_dir, f'model_{i}')
            model.save_model(save_pth)
        

def test(data, device):
    model.to(device)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    input = data[:,0]
    pos_x = 450 - input[:,1]*10.5
    pos_y = 285.7940875237164 +30*10.5 - input[:,0]*10.5
    truth = data[:,1]
    with torch.no_grad():
        predict = model(input)
        test_error = ((predict[:,0] - truth[:,0])**2 + (predict[:, 1] - truth[:, 1])**2).sum() / data.shape[0]
        rule_error = ((pos_x - truth[:,0])**2 + (pos_y - truth[:,1])**2).sum() / data.shape[0]
    print(f'test_error:{test_error.item()}, rule_error:{rule_error.item()}')

class PredictNet(nn.Module):
    """
    网络：根据观察到的点，预测冰壶位置
    """
    def __init__(self):
        super().__init__()
        self.linear_layer = mlp([2]+[64]+[64]+[2], nn.LeakyReLU)

    def forward(self, input):
        out = self.linear_layer(input)
        return out
    
    def save_model(self, pth):

        torch.save(self.state_dict(), pth)
    
    def load_model(self, pth):

        self.load_state_dict(torch.load(pth))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
# 读取完整的数据
for i in range(6):
    load_path = path.join(assets_dir(),'expert_traj/expert_traj_{}.p'.format(i))
    samples = pickle.load(open(load_path, 'rb'))
    if i == 0:
        data = np.array(samples)
    else:
        data = np.append(data, samples, axis=0)
# 分成训练集和测试集
perm = np.arange(data.shape[0])
np.random.shuffle(perm)
shuffle_data = deepcopy(data[perm])
train_data = shuffle_data[:int(0.9*data.shape[0])]
test_data = shuffle_data[int(0.9*data.shape[0]):data.shape[0]]
# 训练
model = PredictNet()
load_dir = path.abspath(path.join(path.dirname(path.abspath(__file__)), '../sp_model'))
load_pth = os.path.join(load_dir, 'model_best')
model.load_model(load_pth)
# train(model, train_data, test_data, epoch=1, batch_size=256, device=device, lr=1e-4, test_interval=10)
test(data, device)
