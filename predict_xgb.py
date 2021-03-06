#coding=utf-8
#########################################################################
# File Name: predict_xgb.py
# Author: guhao
# mail: guhaohit@foxmail.com
# Created Time: 2015年11月30日 星期一 21时42分45秒
#########################################################################

#!/usr/bin/python

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from dataset import Dataset
import xgboost as xgb

def predict():
    with open("models/xgb_71.pkl", "rb") as f:
        model, scaler = pickle.load(f)

    dataset = Dataset.load_pkl("result/all_data.pkl")

    my_pred = []
    def cal_one_user_pred(user_info):
    
        coupon_feats = user_info["coupon_feats"]
        
        pred = np.zeros(len(coupon_feats), dtype=np.float32)
        
        dtest = xgb.DMatrix(scaler.transform(coupon_feats))
        pred = model.predict(dtest)
                    
        scores = zip(pred, user_info["coupon_ids"])
        scores = sorted(scores, key = lambda x: x[0])
        
        coupon_ids = " ".join(map(lambda score: str(score[1]), scores[0:10]))                                                         
        my_pred.append([user_info["user_id"], coupon_ids])
        
    dataset.gen_test_pred(cal_one_user_pred)
    my_pred = sorted(my_pred, key=lambda rec: rec[0])
    fp = open("result/submission_xgb.csv", "w")
    fp.write("USER_ID_hash,PURCHASED_COUPONS\n")
    for pred in my_pred:
        fp.write("%s,%s\n" % (pred[0], pred[1]))                   
        
    fp.close()

predict()
