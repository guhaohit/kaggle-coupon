#coding=utf-8
#########################################################################
# File Name: xgb_train.py
# Author: guhao
# mail: guhaohit@foxmail.com
# Created Time: 2015年11月30日 星期一 20时08分35秒
#########################################################################

#!/usr/bin/python

import xgboost as xgb
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import argparse
import sys
from dataset import Dataset

NEGA_WEIGHT = 2                                                    

def train():
    parser = argparse.ArgumentParser(description='kaggle-coupon-purchase-prediction-train-function')
    parser.add_argument('--seed', '-s', default=71, type=int,
            help='Set the random seed')
    
    parser.add_argument('--validation', '-v', action='store_true',
            help='Set the validation mode')
    args = parser.parse_args()
    model_name = "xgb"
       
    if args.validation:
        dataset = Dataset.load_pkl("result/valid_28.pkl")
        model_name += "_valid28"
    else:
        dataset = Dataset.load_pkl("result/all_data.pkl")
        
    np.random.seed(args.seed)
    scaler = StandardScaler()
    # 进行归一化
    x,y = dataset.gen_train_data(num_nega=NEGA_WEIGHT)
    scaler.fit(x)

    x = scaler.transform(x)

    dtrain = xgb.DMatrix(x, label = y)
    param = {'max_depth':5, 'eta':0.1, 'silent':1, 'objective':'binary:logistic' }
    num_round = 10
    model = xgb.train(param, dtrain, num_round)

    with open("models/{}_{}.pkl".format(model_name, args.seed), "wb") as f:
        pickle.dump([model, scaler], f)


train()   
