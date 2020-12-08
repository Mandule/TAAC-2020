"""
    从 log.pkl 文件中，提取所有用户的行为序列，生成 seqs.pkl 文件。
    seqs.pkl：包含了所有用户的行为序列，训练集用户label。
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

from joblib import Parallel,delayed
tqdm.pandas(desc='pandas bar')

# 加载数据
log = pd.read_pickle('./data/log.pkl')
ad = pd.read_pickle('./data/ad.pkl')

# 将 log，ad 根据 creaive 键合并
log = pd.merge(log, ad, on='creative', how='left')

def func(user_log):
    user_log = user_log.sort_values('time')
    return pd.DataFrame({
        'time' : ' '.join(user_log['time'].map(str).values.tolist()),
        'click' : ' '.join(user_log['click'].map(str).values.tolist()),
        'creative' : ' '.join(user_log['creative'].map(str).values.tolist()),
        'ad' : ' '.join(user_log['ad'].map(str).values.tolist()),
        'product' : ' '.join(user_log['product'].map(str).values.tolist()),
        'category' : ' '.join(user_log['category'].map(str).values.tolist()),
        'ader' : ' '.join(user_log['ader'].map(str).values.tolist()),
        'industry' : ' '.join(user_log['industry'].map(str).values.tolist())
    }, index=[user_log['user'].iloc[0]])

def apply_parallel(df_grouped, func, n_jobs):
    results = Parallel(n_jobs=n_jobs, verbose=1)(delayed(func)(group) for name,group in df_grouped)
    return pd.concat(results)

# 提取每个用户的行为序列
seqs = apply_parallel(log.groupby('user'), func, 32)

seqs.index.name = 'user'
seqs = seqs.reset_index()

# 加载训练集 label，并将 age 和 gender 统一为 20 分类标签
train = pd.read_pickle('./data/train.pkl')
train['labels'] = (train.gender - 1) * 10 + train.age

# 将seqs 和 label合并
seqs = pd.merge(seqs, train[['user', 'age', 'gender', 'labels']], on='user', how='left')
seqs.to_pickle('./data/seqs.pkl')

