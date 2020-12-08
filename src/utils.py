import os
import math
import random
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch import nn
from torch.nn import utils, init
from torch.utils import data
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required

# 设置所有随机种子
def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

# 获取日志类
def get_logger(filename, name=None, level=1):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[level])
    
    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger

# 早停
class EarlyStopping:
    def __init__(self, early_stop_round, logger, task=None):
        self.early_stop_round = early_stop_round
        self.logger = logger
        self.task = task
        
        self.counter = 0
        self.best_acc = float('-inf')
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss, val_acc, model):
        if val_loss > self.best_loss and val_acc < self.best_acc:
            self.counter += 1
            self.logger.warning('early stop count {} out of {} , best_val_acc {:.4f}, best_val_loss {:.4f}'
                                .format(self.counter, self.early_stop_round, self.best_acc, self.best_loss))
            if self.counter >= self.early_stop_round:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_acc = val_acc
            self.best_loss = val_loss
            torch.save(model.state_dict(), '{}_checkpoint.pt'.format(self.task))

# 标签平滑
class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

# 加载用于训练和预测的所有数据
def load_data(cols, embed_dir, seqs_file, feas_file):
    seqs = pd.read_pickle(seqs_file)
    feas = pd.read_pickle(feas_file)
    vocabs = [np.load('{}/{}_vocab.npy'.format(embed_dir, col), allow_pickle=True).item() for col in cols]
    embeddings = [torch.from_numpy(np.load('{}/{}.npy'.format(embed_dir, col))) for col in cols]
    embed_size = sum([e.shape[1] for e in embeddings])
    fea_size = feas.shape[1]
    
    for i in range(len(cols)):
        col = cols[i]
        vocab = vocabs[i]
        tqdm.pandas(desc=col)
        seqs[col] = seqs[col].progress_apply(lambda seq: np.array([vocab[w] for w in seq.split()]))
    
    # 测试集
    sub_seqs = seqs[seqs['labels'].isna()][cols].values
    sub_users = seqs[seqs['labels'].isna()]['user'].values
    sub_feas = feas.loc[sub_users].values
    
    # 训练集
    train_val_seqs = seqs[~seqs['labels'].isna()][cols].values
    train_val_users = seqs[~seqs['labels'].isna()]['user'].values
    train_val_labels = seqs[~seqs['labels'].isna()]['labels'].values - 1
    train_val_feas = feas.loc[train_val_users].values
    
    train_val_data = {
        'seqs' : train_val_seqs,
        'feas' : train_val_feas,
        'labels' : train_val_labels,
        'users' : train_val_users,
        'num' : train_val_labels.shape[0]
    }
    
    sub_data = {
        'seqs' : sub_seqs,
        'feas' : sub_feas,
        'users' : sub_users,
        'num' : sub_users.shape[0]
    }
    
    return train_val_data, sub_data, embeddings, embed_size, fea_size

# Ranger: RAdam + LookAhead 优化器
class Ranger(Optimizer):
    def __init__(self, params, lr=1e-3,                       # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,           # Ranger options
                 betas=(.95, 0.999), eps=1e-8, weight_decay=0.01,  # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 use_gc=True, gc_conv_only=False):
        
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')
        
        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.
        
        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold
        
        # look ahead params
        
        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.use_gc = use_gc

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_gradient_threshold == 1):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_gradient_threshold == 3):
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        #loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state['step'] += 1
                
                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                
                buffered = self.radam_buffer[int(state['step'] % 10)]
                
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                        state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
                
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay']
                                     * group['lr'], p_data_fp32)
                
                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size *
                                         group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                
                p.data.copy_(p_data_fp32)
                
                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(self.alpha, p.data - slow_p)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)
                
        return loss