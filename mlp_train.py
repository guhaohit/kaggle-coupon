#coding=utf-8
#########################################################################
# File Name: mlp_train.py
# Author: guhao
# mail: guhaohit@foxmail.com
# Created Time: 2015年11月14日 星期六 19时04分41秒
#########################################################################

#!/usr/bin/python
import numpy as np                                                      
import pickle
from sklearn.preprocessing import StandardScaler
from chainer import Variable
import argparse
import sys
from mlp3 import MLP3
from dataset import Dataset

NEGA_WEIGHT = 2
EPOCH_NUM = 120
BATCH_SIZE = 128

def train():
    parser = argparse.ArgumentParser(description='kaggle-coupon-purchase-prediction-train-function')
    parser.add_argument('--seed', '-s', default=71, type=int,
            help='Set the random seed')
    parser.add_argument('--validation', '-v', action='store_true',
            help='Set the validation mode')
    args = parser.parse_args()
    model_name = "mlp"

    if args.validation:
        dataset = Dataset.load_pkl("result/valid_28.pkl")
        model_name += "_valid28"
    else:
        dataset = Dataset.load_pkl("result/all_data.pkl")

    np.random.seed(args.seed)

    model = MLP3({"input": dataset.dim(),
                  "lr": 0.01,
                  "h1": 512,
                  "h2": 32,
                  "dropout1": 0.5,
                  "dropout2": 0.1,
                  })

    scaler = StandardScaler()
    # 进行归一化
    x,y = dataset.gen_train_data(num_nega=NEGA_WEIGHT)
    scaler.fit(x)

    # 开始训练
    print('***** now training... *****\n')
    for epoch in xrange(1, EPOCH_NUM+1):
        print('***** epoch {}/{} *****\n'.format(epoch, EPOCH_NUM))
        if epoch == 100:
            model.learning_rate_decay(0.5)

        # 重新进行样本采集，作为训练集
        x,y = dataset.gen_train_data(num_nega=NEGA_WEIGHT)
        x = scaler.transform(x)

        # 使用模型进行训练
        model.train(x, y, batchsize=BATCH_SIZE, verbose=True)

        # 如果是验证集，计算map@k值
        if args.validation and epoch % 10 == 0:
            now_mapk = calculate_map_k(model, dataset, scaler, k=10)
            print("valid MAP@{}: {}\n".format(k, now_mapk / len(dataset.users))),
            sys.stdout.flush()

    with open("models/{}_{}.pkl".format(model_name, args.seed), "wb") as f:
        pickle.dump([model, scaler], f)



# 计算map@k的函数
def calculate_map_k(model, dataset, scaler, k=10):
    def one_mapk(user_info):
        purchased_ids = dataset.users[user_info["user_id"]]["valid_coupon_ids"]

        mapk = 0.0
        if len(purchased_ids) > 0:
            pred = model.predict(scaler.transform(user_info["coupon_feats"]))
            results = zip(pred, user_info["coupon_ids"])
            results = sorted(results, key=lambda x: -x[0])
            correct = 0.0
            for j in xrange(min(k, len(results))):
                if (results[j][1] in purchased_ids):
                    correct += 1.0
                    mapk += (correct / (j + 1.0))
            mapk = mapk / min(k, len(purchased_ids))
        return mapk
    
    return dataset.calculate_valid(one_mapk)
    
train()
