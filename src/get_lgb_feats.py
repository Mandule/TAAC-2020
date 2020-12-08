#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import lightgbm as lgb
from lightgbm import LGBMClassifier

from scipy.sparse import hstack, vstack
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

#获取完整用户行为数据
fp = './data/seqs.pkl'
behaviour = pd.read_pickle(fp)

#tfidf特征
def featureTFIDF(data,dimension,num):
    #data是行为数据,dimension是要提取的维度
    tfidf = TfidfVectorizer(max_features=num)
    tfidf.fit(data[dimension])
    X_train = tfidf.transform(data[~data['labels'].isna()][dimension])
    X_test = tfidf.transform(data[data['labels'].isna()][dimension])
    return X_train,X_test

#对用户行为序列进行creative特征提取count
def featureCount(data,dimension,num):
    #data是行为数据,dimension是要提取的维度
    count = CountVectorizer(max_features=num)
    count.fit(data[dimension])
    X_train = count.transform(data[~data['labels'].isna()][dimension])
    X_test = count.transform(data[data['labels'].isna()][dimension])
    return X_train,X_test

#融合tfidf
Creative_train,Creative_test = featureTFIDF(behaviour,'creative',10000)
Ad_train,Ad_test = featureTFIDF(behaviour,'ad',10000)
Product_train,Product_test = featureTFIDF(behaviour,'product',10000)
Category_train,Category_test = featureTFIDF(behaviour,'category',18)
Ader_train,Ader_test = featureTFIDF(behaviour,'ader',10000)
Indus_train,Indus_test = featureTFIDF(behaviour,'industry',336)
#融合一下
tfidf_train = hstack([Creative_train,Ad_train,Product_train,Category_train,Ader_train,Indus_train])
tfidf_test = hstack([Creative_test,Ad_test,Product_test,Category_test,Ader_test,Indus_test])

#融合count
Creative_train,Creative_test = featureCount(behaviour,'creative',10000)
Ad_train,Ad_test = featureCount(behaviour,'ad',10000)
Product_train,Product_test = featureCount(behaviour,'product',10000)
Category_train,Category_test = featureCount(behaviour,'category',18)
Ader_train,Ader_test = featureCount(behaviour,'ader',10000)
Indus_train,Indus_test = featureCount(behaviour,'industry',336)
# # 融合一下
Count_train = hstack([Creative_train,Ad_train,Product_train,Category_train,Ader_train,Indus_train])
Count_test = hstack([Creative_test,Ad_test,Product_test,Category_test,Ader_test,Indus_test])
Count_train = Count_train.astype(np.float64)
Count_test = Count_test.astype(np.float64)

#将多种特征混合
def mix_feature():
    # 将tfidf和count的特征进行融合
    X_train = sparse.hstack([tfidf_train,Count_train])
    X_test = sparse.hstack([tfidf_test,Count_test])

    X_train = X_train.tocsr()
    X_test = X_test.tocsr()
    
    return X_train,X_test

#最后融合成完整的特征
train_feat,test_feat = mix_feature()

#现在预测的是gender/age/混合标签

label = 'labels'
user = np.array(behaviour[~behaviour[label].isna()][label])

def evaluate(x, y, model, info):
    # 结果评估函数，返回准确率
    y_pred = model.predict(x)
    y_true = y
    age_pred = [y_pred[i]%10 for i in range(len(y_pred))]
    age_true = [y_true[i]%10 for i in range(len(y_true))]
    gender_pred = [int(y_pred[i]/10) for i in range(len(y_pred))]
    gender_true = [int(y_true[i]/10) for i in range(len(y_true))]
    acc1 = accuracy_score(age_pred,age_true)
    acc2 = accuracy_score(gender_pred,gender_true)
    acc = acc1 + acc2
    return acc

#设置一下参数
n_folds = 5
folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1014)

########################### 5_folds_lgb ################################
stack_train = np.zeros((train_feat.shape[0], 20))
stack_test = np.zeros((test_feat.shape[0], 20))
#存储准确率score
score = np.zeros(n_folds)

for i, (tr, va) in enumerate(folds.split(train_feat,user)):
    # 设置lgb的参数
    clf = LGBMClassifier(
            num_iterations=10000,
            learning_rate=0.01,
            num_leaves=63,
            num_class=20,
            objective='multiclass',
            feature_fraction = 0.6,
            bagging_fraction = 0.6,
            is_sparse=True,
            lambda_l1 = 0.5)
    #开始训练
    clf.fit(train_feat[tr], user[tr],
            eval_set=[(train_feat[va], user[va]), (train_feat[tr], user[tr])],
            early_stopping_rounds = 10, verbose = 100)
    
    #输出该折训练结果
    score[i] = evaluate(train_feat[va], user[va], clf, 'val')
    evaluate(train_feat[tr], user[tr], clf, 'train')
    
    #保存该折结果
    score_va = clf.predict_proba(train_feat[va])
    score_te = clf.predict_proba(test_feat)*score[i]
    
    stack_train[va] = score_va
    stack_test += score_te

stack_test /= sum(score)

stack = np.vstack([stack_train, stack_test])
stack = pd.DataFrame(stack, index=behaviour['user'])
stack.to_pickle('./data/lgb_feas.pkl')