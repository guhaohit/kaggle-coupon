#coding=utf-8
#########################################################################
# File Name: predict.py
# Author: guhao
# mail: guhaohit@foxmail.com
# Created Time: 2015年11月15日 星期日 17时51分05秒
#########################################################################

#!/usr/bin/python
import numpy as np                                                      
import pickle
from mlp3 import MLP3
from sklearn.preprocessing import StandardScaler
from dataset import Dataset

def predict():
    models = []
    scalers = []
    my_pred = []
    dataset = Dataset.load_pkl("result/all_data.pkl")
    for i in [71, 72, 73, 74]:
        model_name = "mlp_{}.pkl".format(i)
        print("load " + model_name)
        with open("models/" + model_name, "rb") as f:
            model, scaler = pickle.load(f)
            models.append(model)
            scalers.append(scaler)

    def cal_one_user_pred(user_info):
        coupon_feats = user_info["coupon_feats"]
        pred = np.zeros(len(coupon_feats), dtype=np.float32)

        for i, now_model in enumerate(models):
            pred += now_model.predict(scalers[i].transform(coupon_feats))
        pred /= len(models)

        scores = zip(pred, user_info["coupon_ids"])
        scores = sorted(scores, key = lambda x: -x[0])

        coupon_ids = " ".join(map(lambda score: str(score[1]), scores[0:10]))
        my_pred.append([user_info["user_id"], coupon_ids])

    dataset.gen_test_pred(cal_one_user_pred)
    my_pred = sorted(my_pred, key=lambda rec: rec[0])
    fp = open("result/submission_mlp.csv", "w")
    fp.write("USER_ID_hash,PURCHASED_COUPONS\n")
    for pred in my_pred:
        fp.write("%s,%s\n" % (pred[0], pred[1]))
    fp.close()

predict()
