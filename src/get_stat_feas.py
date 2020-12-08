import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
tqdm.pandas(desc='pandas bar')

log = pd.read_pickle('./data/log.pkl')
ad = pd.read_pickle('./data/ad.pkl')
del log['train']

log = pd.merge(log, ad, how='left', on='creative')

log['day'] = log['time'].progress_apply(lambda time: time%7)
log['month'] = pd.cut(log['time'], 3, labels=[0, 1, 2]).astype(int)

def get_feas(user_log):
    feas = {}
    # 总点击数
    feas['total'] = user_log.shape[0]
    # 点击数星期
    day_count = user_log['day'].value_counts()
    for i in range(7):
        feas['day_{}'.format(i)] = 0.0
    for i in day_count.index:
        feas['day_{}'.format(i)] = day_count[i] / feas['total']
    # 点击数月分布
    month_count = user_log['month'].value_counts()
    for i in range(3):
        feas['month_{}'.format(i)] = 0.0
    for i in month_count.index:
        feas['month_{}'.format(i)] = month_count[i] / feas['total']
    # nunique
    for col in ['time', 'creative', 'ad', 'product', 'category', 'ader', 'industry']:
        feas['{}_nunique'.format(col)] = user_log[col].nunique()
    # 平均每日点击数
    feas['click_day'] = feas['total'] / feas['time_nunique']
    # 平均每日nuniqee
    for col in ['creative', 'ad', 'product', 'category', 'ader', 'industry']:
        feas['{}_nunique_day'.format(col)] = feas['{}_nunique'.format(col)] / feas['time_nunique']
    return pd.Series(feas)

feas = log.groupby('user').progress_apply(get_feas)
feas = pd.DataFrame(MinMaxScaler().fit_transform(feas), index=feas.index)

feas.to_pickle('./data/stat_feas.pkl')