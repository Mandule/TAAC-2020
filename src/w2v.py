import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from tqdm import tqdm
tqdm.pandas(desc='pandas bar')

# 加载数据
seqs = pd.read_pickle('./data/seqs.pkl')

cols = ['time', 'click', 'creative', 'ad', 'product', 'category', 'ader', 'industry']
sizes = [32, 32, 128, 128, 128, 32, 128, 64]

# 训练 8 类序列的 word2vec 向量
for i in range(len(cols)):
    col = cols[i]
    size = sizes[i]
    tqdm.pandas(desc=col)
    corpus = seqs[col].progress_apply(lambda s: [w for w in s.split() if w != '0']).values
    print('start training {} w2v {}'.format(col, size))
    model = Word2Vec(corpus, size=size, window=10, min_count=1, workers=36, sg=1)
    vocab = {'0' : 0}
    for key, value in model.wv.vocab.items():
        vocab[key] = value.index + 1
    pad = np.array([[0.0]*size]).astype(np.float32)
    vec = np.concatenate([pad, model.wv.vectors]).astype(np.float32)
    
    np.save('./data/w2v/{}'.format(col), vec)
    np.save('./data/w2v/{}_vocab'.format(col), vocab)
