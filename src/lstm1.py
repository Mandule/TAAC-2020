"""
    lstm 模型训练。
    随机 shuffle + mutil-sample-dropout。
    收敛速度显著提升。
"""
import math
import random
import logging

from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn import utils, init
from torch.utils import data
import torch.optim as optim
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim.optimizer import Optimizer, required

from sklearn.model_selection import StratifiedKFold

from utils import *

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 自定义训练集
class SeqDataSet(data.Dataset):
    def __init__(self, multi_seqs, feas, labels, num_seqs, max_len, tag):
        self.multi_seqs = multi_seqs
        self.feas = feas
        self.labels = labels
        self.tag = tag
        
        self.num_seqs = num_seqs
        self.max_len = max_len
        
    def __getitem__(self, index):
        multi_seq = self.multi_seqs[index]
        fea = self.feas[index]
        label = self.labels[index]
        return multi_seq, fea, label
        
    def __len__(self):
        return len(self.labels)
    
    def collate_fn(self, batch_data):
        batch_data.sort(key=lambda data: data[0][0].shape[0], reverse=True)
        multi_seqs = [[] for i in range(self.num_seqs)]
        feas = []
        labels = []
        lens = []
        
        for data in batch_data:
            multi_seq = data[0]
            lens.append(min(multi_seq[0].shape[0], self.max_len))
            index = np.arange(multi_seq[0].shape[0])
            random.shuffle(index)
            index = index[:self.max_len]
            for i in range(self.num_seqs):
                multi_seqs[i].append(torch.LongTensor(multi_seq[i][index]))
            feas.append(data[1])
            labels.append(data[2])
        
        # multi_seqs [num_seqs, (batch_size, len)]
        # feas [batch_size, fea_size]
        # lens [batch_size]
        # labels [batch_size]
        multi_seqs = torch.stack([rnn.pad_sequence(seqs, batch_first=True, padding_value=0) for seqs in multi_seqs])
        # multi_seqs [num_seqs, batch_size, max_len]
        multi_seqs = multi_seqs.permute(1, 0, 2)
        # multi_seqs [batch_size, num_seqs, max_len]
        
        feas = torch.FloatTensor(feas)
        labels = torch.LongTensor(labels)
        lens = torch.IntTensor(lens)
        
        return multi_seqs, feas, lens, labels

# embedding 层，w2v向量转换和拼接
class embedNet(nn.Module):
    def __init__(self, embeddings):
        super(embedNet, self).__init__()
        embed_layers = [nn.Embedding.from_pretrained(embedding, padding_idx=0) for embedding in embeddings]
        self.num_seqs = len(embed_layers)
        self.embed_layers = nn.ModuleList(embed_layers)
        
    def forward(self, users_seqs):
        #users_seqs  [batch_size, num_seqs, max_len]
        users_seqs = users_seqs.permute(1, 0, 2)
        embeddings = [self.embed_layers[i](users_seqs[i]) for i in range(self.num_seqs)]
        embeddings = torch.cat(embeddings, dim=2)
        #embeddings [batch_size, max_len, embed_size]
        return embeddings

# lstm 层
class lstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_out):
        super(lstmNet, self).__init__()
        self.LSTMLayer = nn.LSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 dropout=drop_out,
                                 bidirectional=True,
                                 batch_first=True)
        self.init_params()
        
    def forward(self, embeddings, lens):
        # embeddings [batch_size, max_len, embed_size]
        x = rnn.pack_padded_sequence(embeddings, lens, batch_first=True)
        x, (h, c) = self.LSTMLayer(x)
        x, _ = rnn.pad_packed_sequence(x, batch_first=True, padding_value=0.0)
        # x [batch_size, max_len, 2 * hidden_size]
        x = torch.transpose(x, 1, 2)
        # x [batch_size, 2 * hidden_sizen, max_len]
        h = F.max_pool1d(x, x.shape[-1]).squeeze()
        # h [batch_size, 2 * hidden_size]
        return h
        
    def init_params(self):
        for layer in range(len(self.LSTMLayer.all_weights)):
            init.orthogonal_(self.LSTMLayer.all_weights[layer][0])
            init.orthogonal_(self.LSTMLayer.all_weights[layer][1])
            init.zeros_(self.LSTMLayer.all_weights[layer][2])
            init.zeros_(self.LSTMLayer.all_weights[layer][3])

# 分类全连接层
class classfiyNet(nn.Module):
    def __init__(self, input_size, num_labels, drop_out, num_drop):
        super(classfiyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.mutil_dropout1 = nn.ModuleList([
            nn.Dropout(drop_out) for _ in range(num_drop)
        ])
        self.fc2 = nn.Linear(input_size // 2, input_size // 4)
        self.mutil_dropout2 = nn.ModuleList([
            nn.Dropout(drop_out) for _ in range(num_drop)
        ])
        self.fc3 = nn.Linear(input_size // 4, num_labels)
    
    def forward(self, x):
        h = self.fc1(x)
        h = F.leaky_relu(h, inplace=True)
        if self.training:
            hs = []
            # hs num_drop
            for dropout in self.mutil_dropout1:
                hs.append(F.leaky_relu(self.fc2(dropout(h)), inplace=True))
            
            ys = []
            # ys num_drop^2
            for dropout in self.mutil_dropout2:
                for h in hs:
                    ys.append(self.fc3(dropout(h)))
            
            return ys
        else:
            h = self.fc2(h)
            h = F.leaky_relu(h, inplace=True)
            y = self.fc3(h)
            
            return y
    
    def init_params(self):
        for name, w in self.fc1.named_parameters():
            if 'weight' in name and w.dim() > 1:
                init.xavier_normal_(w)
            else:
                init.zeros_(w)
        for name, w in self.fc2.named_parameters():
            if 'weight' in name and w.dim() > 1:
                init.xavier_normal_(w)
            else:
                init.zeros_(w)
        for name, w in self.fc3.named_parameters():
            if 'weight' in name and w.dim() > 1:
                init.xavier_normal_(w)
            else:
                init.zeros_(w)

# 完整的网路结构
class Net(nn.Module):
    def __init__(self, embed_size, fea_size, hidden_size, num_layers, drop_out, num_drop):
        super(Net, self).__init__()
        self.lstm_layer = lstmNet(embed_size, hidden_size, num_layers, drop_out)
        self.fc_feas = nn.Sequential(
            nn.Linear(fea_size, hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.fc_embed = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.LeakyReLU(inplace=True),
        )
        self.init_params()
        self.fc_output = classfiyNet(4 * hidden_size, 20, drop_out, num_drop)
    
    def forward(self, embeddings, feas, lens):
        # lens [batch_size]
        # feas [batch_size, fea_size]
        # embeddings [batch_size, max_len, embed_size]
        lstm_output = self.lstm_layer(embeddings, lens)
        # lstm_output [batch_size, 2 * hidden_size]
        embeddings = embeddings.permute(0, 2, 1)
        embed_output = F.max_pool1d(embeddings, embeddings.shape[-1]).squeeze()
        embed_output = self.fc_embed(embed_output)
        # embed_output [batch_size, hidden_size]
        feas_output = self.fc_feas(feas)
        # feas_output [batch_size, hidden_size]
        h = torch.cat([embed_output, lstm_output, feas_output], dim=1)
        # h [batch_size, 4 * hidden_size]
        y = self.fc_output(h)
        # y  num_drop * [batch_size, 20]
        return y
    
    def init_params(self):
        for name, w in self.fc_feas.named_parameters():
            if 'weight' in name and w.dim() > 1:
                init.xavier_normal_(w)
            else:
                init.zeros_(w)
        for name, w in self.fc_embed.named_parameters():
            if 'weight' in name and w.dim() > 1:
                init.xavier_normal_(w)
            else:
                init.zeros_(w)

def train(params):
    logger = get_logger('{}.log'.format(params['task']), '{}_logger'.format(params['task']))
    logger.info('start {}'.format(params['task']))
    
    set_all_seed(params['seed'])
    
    for key, value in params.items():
        logger.info('{} : {}'.format(key, value))
    
    logger.info('loading seqs, feas and w2v embeddings ...')
    train_val_data, sub_data, embeddings, embed_size, fea_size = load_data(params['cols'], params['embed_dir'], params['seqs_file'], params['feas_file'])
    
    logger.info('embed_size : {} | fea_size : {}'.format(embed_size, fea_size))
    batch_size = params['batch_size']
    sub_dataset = SeqDataSet(sub_data['seqs'],
                             sub_data['feas'],
                             sub_data['users'], len(params['cols']), params['max_len'], 'sub')
    sub_loader = data.DataLoader(sub_dataset, batch_size * 10, shuffle=False, collate_fn=sub_dataset.collate_fn, pin_memory=True)
    
    sub = np.zeros(shape=(sub_data['num'], 20))
    sub = pd.DataFrame(sub, index=sub_data['users'])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=params['seed'])
    
    for i, (train_idx, val_idx) in enumerate(skf.split(train_val_data['feas'], train_val_data['labels'])):
        logger.info('------------------------------------------{} fold------------------------------------------'.format(i))
        train_dataset = SeqDataSet(train_val_data['seqs'][train_idx],
                                   train_val_data['feas'][train_idx],
                                   train_val_data['labels'][train_idx], len(params['cols']), params['max_len'], 'train')
        train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, pin_memory=True)
        
        val_dataset = SeqDataSet(train_val_data['seqs'][val_idx],
                                 train_val_data['feas'][val_idx],
                                 train_val_data['labels'][val_idx], len(params['cols']), params['max_len'], 'val')
        val_loader = data.DataLoader(val_dataset, batch_size * 10, shuffle=False, collate_fn=val_dataset.collate_fn, pin_memory=True)
        
        logger.info('train samples : {} | val samples : {} | sub samples : {}'.format(len(train_idx), len(val_idx), sub_data['num']))
        logger.info('loading net ...')
        
        embed_net = embedNet(embeddings).cuda()
        net = Net(embed_size, fea_size, params['hidden_size'], params['num_layers'], params['drop_out'], params['num_drop']).cuda()
        
        #optimizer = Ranger(params=net.parameters(), lr=params['lr'])
        optimizer = optim.AdamW(params=net.parameters(), lr=params['lr'])
        scheduler = StepLR(optimizer, step_size=2, gamma=params['gamma'])
        #scheduler = CosineAnnealingLR(optimizer, T_max=params['num_epochs'])
        loss_func = CrossEntropyLabelSmooth(20, params['label_smooth'])
        #loss_func = nn.CrossEntropyLoss()
        
        earlystop = EarlyStopping(params['early_stop_round'], logger, params['task'] + str(i))
        
        for epoch in range(params['num_epochs']):
            train_loss, val_loss = 0.0, 0.0
            train_age_acc, val_age_acc = 0.0, 0.0
            train_gender_acc, val_gender_acc = 0.0, 0.0
            train_acc, val_acc = 0.0, 0.0
            
            n, m = 0, 0
            lr_now = scheduler.get_last_lr()[0]
            logger.info('--> [Epoch {:02d}/{:02d}] lr = {:.7f}'.format(epoch, params['num_epochs'], lr_now))
            
            # 训练模型
            net.train()
            for seqs, feas, lens, labels in tqdm(train_loader, desc='[Epoch {:02d}/{:02d}] Train'.format(epoch, params['num_epochs'])):
                seqs = seqs.cuda()
                feas = feas.cuda()
                lens = lens.cuda()
                labels = labels.cuda()
                
                logits = net(embed_net(seqs), feas, lens)
                loss = sum([loss_func(logit, labels) for logit in logits]) / len(logits)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                logits = sum(logits) / len(logits)
                
                train_loss += loss.detach() * labels.shape[0]
                train_age_acc += (logits.argmax(dim=1).detach() % 10 == labels % 10).sum()
                train_gender_acc += (logits.argmax(dim=1).detach() // 10 == labels // 10).sum()
                
                n += lens.shape[0]
            
            scheduler.step()
            
            train_loss = (train_loss / n).item()
            train_age_acc = (train_age_acc / n).item()
            train_gender_acc = (train_gender_acc / n).item()
            train_acc = train_age_acc + train_gender_acc
            
            # 预测验证集
            net.eval()
            with torch.no_grad():
                for seqs, feas, lens, labels in tqdm(val_loader, desc='[Epoch {:02d}/{:02d}]  Val '.format(epoch, params['num_epochs'])):
                    seqs = seqs.cuda()
                    feas = feas.cuda()
                    lens = lens.cuda()
                    labels = labels.cuda()
                    
                    logits = net(embed_net(seqs), feas, lens)
                    loss = loss_func(logits, labels)
                    
                    val_loss += loss.detach() * labels.shape[0]
                    val_age_acc += (logits.argmax(dim=1) % 10 == labels % 10).sum()
                    val_gender_acc += (logits.argmax(dim=1).detach() // 10 == labels // 10).sum()
                    
                    m += lens.shape[0]
                
                val_loss = (val_loss / m).item()
                val_age_acc = (val_age_acc / m).item()
                val_gender_acc = (val_gender_acc / m).item()
                val_acc = val_age_acc + val_gender_acc
            
            logger.info('train_loss {:.5f} | train_gender_acc {:.5f} | train_age_acc {:.5f} | train_acc {:.5f} | val_loss {:.5f} | val_gender_acc {:.5f} | val_age_acc {:.5f} | val_acc {:.5f}'
                        .format(train_loss, train_gender_acc, train_age_acc, train_acc, val_loss, val_gender_acc, val_age_acc, val_acc))
            
            # 早停
            earlystop(val_loss, val_acc, net)
            if earlystop.early_stop:
                break
        
        break 
        net.load_state_dict(torch.load('{}_checkpoint.pt'.format(params['task']+str(i))))
        logger.info('predicting sub ...')
        net.eval()
        with torch.no_grad():
            for it in range(10):
                probs = []
                users = []
                for seqs, feas, lens, ids in tqdm(sub_loader, desc='predict_{}'.format(it)):
                    seqs = seqs.cuda()
                    feas = feas.cuda()
                    lens = lens.cuda()
                    
                    logits = net(embed_net(seqs), feas, lens)
                    logits = F.softmax(logits, dim=1)
                    
                    probs.append(logits)
                    users.append(ids)
                    
                probs = torch.cat(probs).cpu().numpy()
                users = torch.cat(users).numpy()
                sub += pd.DataFrame(probs, users)
            sub = sub / 10
            
    return sub 

if __name__ == '__main__':
    
    params = {
        'seed' : 2020,
        'task' : 'lstm1',
        'embed_dir' : './data/w2v',
        'seqs_file' : './data/seqs.pkl',
        'feas_file' : './data/feas.pkl',
        'cols' : ['time', 'click', 'creative', 'ad', 'product', 'ader', 'industry', 'category'],
        'batch_size' : 256,
        'max_len' : 256,
        'hidden_size' : 256,
        'num_layers' : 2,
        'drop_out' : 0.2,
        'num_drop' : 2,
        'label_smooth' : 0.1,
        'num_epochs' : 50,
        'early_stop_round' : 3,
        'lr' : 1e-3,
        'gamma' : 0.75,
    }
    
    sub = train(params)
    sub_split = pd.DataFrame(np.zeros((sub.shape[0], 12)), index=sub.index)
    sub_split.loc[:,0] = sub.loc[:,:9].sum(axis=1)
    sub_split.loc[:,1] = sub.loc[:,10:].sum(axis=1)
    for i in range(10):
        sub_split.loc[:,i+2] = sub.loc[:,[i,i+10]].sum(axis=1)
    sub_split.to_pickle('./torch_{}_sub.pkl'.format(params['task']))