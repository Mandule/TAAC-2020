"""
    将原始 test, train 文件整理合并成三个文件 log.pkl, ad.pkl, train.pkl 文件。
    log.pkl：合并了所有 log 文件，并去除了一些异常日志和异常用户产生的日志。
    ad.pkl：合并了所有 ad 文件
    train.pkl：合并了所有 train 文件
"""
import pandas as pd

# 加载 test，train 的 ad 文件
ad_test = pd.read_csv('./data/test/ad.csv')
ad_train_0 = pd.read_csv('./data/train_preliminary/ad.csv')
ad_train_1 = pd.read_csv('./data/train_semi_final/ad.csv')

# 合并所有 ad
ad = pd.concat([ad_test, ad_train_0, ad_train_1])
ad = ad.drop_duplicates()

# 缺失值用 0 填充
ad['product_id'] = ad['product_id'].apply(lambda x: 0 if x == '\\N' else int(x))
ad['industry'] = ad['industry'].apply(lambda x: 0 if x == '\\N' else int(x))

# 修改列名
ad.columns = ['creative', 'ad', 'product', 'category', 'ader', 'industry']
ad.to_pickle('data/ad.pkl')

# 加载 test，train 的 click_log 文件
log_test = pd.read_csv('./data/test/click_log.csv')
log_train_0 = pd.read_csv('./data/train_preliminary/click_log.csv')
log_train_1 = pd.read_csv('./data/train_semi_final/click_log.csv')

# 合并所有 click_log
log_train_0['train'] = True
log_train_1['train'] = True
log_test['train'] = False
log = pd.concat([log_test, log_train_0, log_train_1])

# 修改列名
log.columns = ['time', 'user', 'creative', 'click', 'train']
log = log.sort_values('user')

# 去除 click >= 10 的点击日志记录
log = log[log['click'] < 10]

train_log = log[log['train']]
test_log = log[~log['train']]

# 统计 训练集，测试集 中用户被记录的点击行为次数
train_freq = train_log['user'].value_counts()
test_freq = test_log['user'].value_counts()

# 将 91 天内 被记录次数超过 1000 次的用户视为异常用户
train_fake = train_freq[train_freq > 1000].index
test_fake = test_freq[test_freq > 1000].index

# 训练集的异常用户直接删除
log = log[~log['user'].isin(train_fake)]

# 测试集用户只保留第一次记录，方便生成 sub 文件
fake_log = log[log['user'].isin(test_fake)]
fake_log = fake_log.drop_duplicates('user')
log = log[~log['user'].isin(test_fake)]
log = pd.concat([log, fake_log]).reset_index(drop=True)

log.to_pickle('data/log.pkl')

# 加载 train 文件
train_0 = pd.read_csv('./data/train_preliminary/user.csv')
train_1 = pd.read_csv('./data/train_semi_final/user.csv')

# 修改列名
train = pd.concat([train_0, train_1])
train.columns = ['user', 'age', 'gender']

train = train.sort_values('user')
train.to_pickle('./data/train.pkl')