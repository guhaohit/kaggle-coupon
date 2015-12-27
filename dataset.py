#coding=utf-8
#########################################################################
# File Name: dataset.py
# Author: guhao
# mail: guhaohit@foxmail.com
# Created Time: 2015年11月12日 星期四 18时57分27秒
#########################################################################

#!/usr/bin/python

import sys
import pickle
import pandas as pd
from exceptions import NotImplementedError
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import os.path as path

class Dataset:
    def __init__(self, datadir="./data"):
        self.datadir = datadir
        self.train_coupon_vec = None
        self.test_coupon_vec = None
        self.train_coupon_df = None
        self.valid_coupon_df = None
        self.test_coupon_df = None
        self.users = None
        self.user_df = None

    def save_pkl(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_pkl(cls, filename):
        with open(filename, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    def load(self, validation_timedelta=None):
        self.__load_coupons(validation_timedelta)
        self.__load_users()
    
    def __load_coupons(self, validation_timedelta):
        train_coupon_df = pd.read_csv(path.join(self.datadir, "coupon_list_train.csv"),parse_dates=["DISPFROM","DISPEND","VALIDFROM"])
        test_coupon_df = pd.read_csv(path.join(self.datadir, "coupon_list_test.csv"),parse_dates=["DISPFROM","VALIDFROM"])

        train_coupon_df["DISPFROM"].fillna(pd.Timestamp("19000101"), inplace=True)
        train_coupon_df = train_coupon_df.sort_values(by="DISPFROM").reset_index(drop=True)
        train_coupon_df["VALIDFROM"].fillna(pd.Timestamp("19000101"), inplace=True)
        test_coupon_df["VALIDFROM"].fillna(pd.Timestamp("19000101"), inplace=True)
        test_coupon_df["DISPFROM"].fillna(pd.Timestamp("19000101"), inplace=True)


        if validation_timedelta:
            max_date = train_coupon_df["DISPFROM"].max()
            valid_start = max_date - validation_timedelta
            valid_coupon_df = train_coupon_df[(train_coupon_df["DISPFROM"] > valid_start)]
            train_coupon_df = train_coupon_df[~ (train_coupon_df["DISPFROM"] > valid_start)]
        else:
            valid_coupon_df = train_coupon_df[np.zeros(len(train_coupon_df), dtype=np.bool)].copy()

        # 将验证集中的outlier去掉
        if len(valid_coupon_df) > 0:
            valid_coupon_df = valid_coupon_df[valid_coupon_df["DISCOUNT_PRICE"] > 100].reset_index(drop=True)
            valid_coupon_df = valid_coupon_df[valid_coupon_df["DISPPERIOD"] <= 20].reset_index(drop=True)

        # 将训练集中的outlier去掉
        train_coupon_df = train_coupon_df[train_coupon_df["DISPPERIOD"] <= 20].reset_index(drop=True)

        # 将coupon数据进行预处理
        self.__coupon_preproc(train_coupon_df)
        if validation_timedelta:
            self.__coupon_preproc(valid_coupon_df)
        self.__coupon_preproc(test_coupon_df)
        
        # 利用pandas-sklearn对coupon的特征进行one-hot编码
        coupon_mapper = DataFrameMapper([
                ('CATEGORY_NAME', LabelBinarizer()),
                ('PRICE_RATE', None),
                ('CATALOG_PRICE_LOG', None),
                ('DISCOUNT_PRICE_LOG', None),
                ('REDUCE_PRICE_LOG', None),
                ('DISPPERIOD_C', LabelBinarizer()),
                ('VALIDPERIOD_NA', LabelBinarizer()),
                ('USABLE_DATE_SUM', None),
                ('LARGE_AREA_NAME', LabelBinarizer()),
                ('PREF_NAME', LabelBinarizer()),
                ('SMALL_AREA_NAME', LabelBinarizer()),
                ('IMMEDIATE_USE_TIME', None),
                ('DISPPERIOD', None),
                ('VALIDPERIOD', None)
                ])
        
        coupon_mapper.fit(pd.concat([train_coupon_df, valid_coupon_df, test_coupon_df]))

        train_coupon_vec = coupon_mapper.transform(train_coupon_df.copy())
        if len(valid_coupon_df) > 0:
            valid_coupon_vec = coupon_mapper.transform(valid_coupon_df.copy())
        else:
            valid_coupon_vec = np.array([])
        test_coupon_vec = coupon_mapper.transform(test_coupon_df.copy())

        self.train_coupon_vec = train_coupon_vec
        self.valid_coupon_vec = valid_coupon_vec
        self.test_coupon_vec = test_coupon_vec
        self.train_coupon_df = train_coupon_df
        self.valid_coupon_df = valid_coupon_df
        self.test_coupon_df = test_coupon_df

    @staticmethod
    def __coupon_preproc(df):
        df["REDUCE_PRICE"] = df["CATALOG_PRICE"] - df["DISCOUNT_PRICE"]
        for key in ["DISCOUNT_PRICE", "CATALOG_PRICE", "REDUCE_PRICE"]:
            df[key + "_LOG"] = np.log(df[key] + 1.0).astype(np.float32)

        df["VALIDPERIOD_NA"] = np.array(pd.isnull(df["VALIDPERIOD"]), dtype=np.int32)
        df["VALIDPERIOD"].fillna(365, inplace=True)
        df["DISPPERIOD_C"] = np.array(df["DISPPERIOD"].clip(0, 8), dtype=np.int32)
        
        use_days = []
        for timedelta in df["VALIDFROM"]-df["DISPFROM"]:
            use_days.append(timedelta.days)
        df["IMMEDIATE_USE_TIME"] = np.array(map(lambda x:max(x,0),use_days), dtype=np.int32)
        df["PRICE_RATE"] = np.array(df.PRICE_RATE, dtype=np.float32)
        
        df["large_area_name"].fillna("NA", inplace=True)
        df["ken_name"].fillna("NA", inplace=True)
        df["small_area_name"].fillna("NA", inplace=True)
        df["LARGE_AREA_NAME"] = df["large_area_name"]
        df["PREF_NAME"] = df["large_area_name"] + ":" + df["ken_name"]
        df["SMALL_AREA_NAME"] = df["large_area_name"] + ":" + df["ken_name"] + ":" + df["small_area_name"]
        df["CATEGORY_NAME"] = df["CAPSULE_TEXT"] + df["GENRE_NAME"]

        usable_dates = ['USABLE_DATE_MON',
                        'USABLE_DATE_TUE',
                        'USABLE_DATE_WED',
                        'USABLE_DATE_THU',
                        'USABLE_DATE_FRI',
                        'USABLE_DATE_SAT',
                        'USABLE_DATE_SUN',
                        'USABLE_DATE_HOLIDAY',
                        'USABLE_DATE_BEFORE_HOLIDAY']
        for key in usable_dates:
            df[key].fillna(0, inplace=True)
            df[key] = np.array(df[key].clip(0,1), dtype = np.int32)

        df["USABLE_DATE_SUM"] = 0
        for key in usable_dates:
            df["USABLE_DATE_SUM"] += df[key]

        cols = df.columns.tolist()
        cols.remove("DISPFROM")
        cols.remove("DISPEND")
        for key in cols:
            df[key].fillna("NA", inplace=True)

    
    def __load_users(self):
        user_df = pd.read_csv(path.join(self.datadir,"user_list.csv"))
        details = pd.read_csv(path.join(self.datadir, "coupon_detail_train.csv"),parse_dates=["I_DATE"])
        details = details.sort_values(by=["I_DATE"]).reset_index(drop=True)

        # user的特征
        user_mapper = DataFrameMapper([
                ('SEX_ID', LabelBinarizer()),
                ('PREF_NAME', LabelBinarizer()),
                ('AGE', None),
                ])
        user_df["PREF_NAME"].fillna("NA", inplace=True)
        user_vec = user_mapper.fit_transform(user_df.copy())

        users = []
        self.train_coupon_df["ROW_ID"] = pd.Series(self.train_coupon_df.index.tolist())
        self.valid_coupon_df["ROW_ID"] = pd.Series(self.valid_coupon_df.index.tolist())
        for i, user in user_df.iterrows():
            coupons = details[details.USER_ID_hash.isin([user["USER_ID_hash"]])]
            train_coupon_data = pd.merge(coupons[["COUPON_ID_hash","ITEM_COUNT","I_DATE"]],self.train_coupon_df,on="COUPON_ID_hash", how='inner',suffixes=["_x",""], copy=False)
            train_coupon_data = train_coupon_data.sort_values(by=["I_DATE"])
            row_ids = train_coupon_data.ROW_ID.unique().tolist()

            valid_coupon_data = pd.merge(coupons[["COUPON_ID_hash","ITEM_COUNT","I_DATE"]],self.valid_coupon_df, on="COUPON_ID_hash",how='inner', suffixes=["_x",""], copy=False)
            valid_coupon_data = valid_coupon_data.sort_values(by=["I_DATE"])
            valid_row_ids = valid_coupon_data.ROW_ID.unique().tolist()

            # 将每个user买过的coupon提取出来,按照购买的时间顺序排列
            users.append({"user": user_vec[i],
                          "coupon_ids": row_ids,
                          "valid_coupon_ids": valid_row_ids})
            if i % 100 == 0:
                print "load users: %d/%d\r" % (i, len(user_df)),
                sys.stdout.flush()

        print "\n",

        self.users = users
        self.user_df = user_df
    
    # 返回我们产生的特征的维度
    def dim(self):
        return len(self.users[0]["user"]) + (len(self.train_coupon_vec[0]) + 2 + 4) + len(self.train_coupon_vec[0])
  
    # 返回我们需要求最大最小值的列
    def __maxmin_columns(self, coupon_ids):                          
        return self.train_coupon_df.ix[
            coupon_ids, ("CATALOG_PRICE","DISCOUNT_PRICE")
            ].as_matrix().astype(np.float32)

    # 产生训练集feature和label
    def gen_train_data(self, num_nega=2, COUPON_DISP_NEAR=400, COUPON_DISP_NEAR_MIN=10):
        feature = []
        label = []
        user_num = 0
        print("generate train data now...\n")
        for user in self.users:
            coupon_ids = np.array(user["coupon_ids"], dtype=np.int32)
            user_coupons = self.train_coupon_vec[coupon_ids]
            maxmin_columns = self.__maxmin_columns(coupon_ids)
            for coupon_num in xrange(len(user_coupons)):
                coupon_id = coupon_ids[coupon_num]
                now_coupon_vec = user_coupons[coupon_num]
                
                # 产生生成负例的列表
                nega_list = range(max(0, coupon_id-COUPON_DISP_NEAR), coupon_id)
                if len(nega_list) < COUPON_DISP_NEAR_MIN:
                    coupon_num += 1
                    continue

                # 不用购买当前coupon之后的购买信息提取特征
                filter_idx = np.ones(user_coupons.shape[0], dtype=np.bool)
                filter_idx[coupon_num:] = False
                filter_idx[coupon_ids == coupon_ids[coupon_num]] = False

                my_user_coupons = user_coupons[filter_idx]
                extract_feature = self.__purchase_history_features(my_user_coupons, maxmin_columns)

                positive_feature = np.hstack((user["user"], extract_feature, now_coupon_vec))
                feature.append(positive_feature)
                label.append([1])

                #随机构造负例训练集
                for i in xrange(num_nega):
                    founded = False
                    for _ in xrange(10):
                        unpurchased_idx = np.random.choice(nega_list, 1)[0]
                        if unpurchased_idx not in user["coupon_ids"]:
                            founded = True
                            break
                    if founded:
                        negative_feature = np.hstack((user["user"], extract_feature, self.train_coupon_vec[unpurchased_idx]))
                        feature.append(negative_feature)
                        label.append([0])
                coupon_num += 1
            user_num += 1

           # if user_num % 100 == 0:
           #     print ("generate train data ... %d/%d\r" % (user_num, len(self.users))),
           #     sys.stdout.flush()
        #print "\n"

        feature = np.array(feature, dtype=np.float32)
        label = np.array(label, dtype=np.int32)

        return feature,label

    def __purchase_history_features(self, user_coupons, maxmin_columns):
        mean_coupon_vec = np.zeros(len(self.train_coupon_vec[0]), dtype=np.float32)
        maxmin_vec = np.zeros((4), dtype=np.float32)
        purchased_num_vec = np.zeros(2, dtype=np.float32)

        if len(user_coupons) > 0:
            mean_coupon_vec = np.array(user_coupons.mean(0), dtype=np.float32)
            
            purchased_num_vec[0] = len(user_coupons)
            purchased_num_vec[1] = np.log(purchased_num_vec[0] + 1.0)

            max_val = maxmin_columns.max(0)
            min_val = maxmin_columns.min(0)
            maxmin_vec[0] = max_val[0]
            maxmin_vec[1] = min_val[0]
            maxmin_vec[2] = max_val[1]
            maxmin_vec[3] = min_val[1]

        return np.hstack((mean_coupon_vec, maxmin_vec, purchased_num_vec))

    # 通过传入计算函数来计算测试集的概率
    def gen_test_pred(self, cal_function):
        for k, user in enumerate(self.users):
            user_id = self.user_df["USER_ID_hash"][k]
            user_coupons = self.train_coupon_vec[user["coupon_ids"]]
            maxmin_columns = self.__maxmin_columns(user["coupon_ids"])
            hist_feat = self.__purchase_history_features(user_coupons, maxmin_columns)
            feats = np.empty((len(self.test_coupon_vec),
                              len(user["user"]) + len(hist_feat) + len(self.test_coupon_vec[0])),dtype=np.float32)
            coupon_ids = []
            for i in xrange(len(self.test_coupon_vec)):
                coupon_id = self.test_coupon_df["COUPON_ID_hash"][i]
                feats[i][:] = np.hstack((user["user"], hist_feat, self.test_coupon_vec[i]))
                coupon_ids.append(coupon_id)

            cal_function({"user_id": user_id, "coupon_ids": coupon_ids, "coupon_feats": feats})
            if k % 100 == 0:
                print ("each test .. %d/%d\r" % (k, len(self.users))),
                sys.stdout.flush()
        
        print "\n",


    # 通过传入计算效率的函数来计算测试集的误差率
    def calculate_valid(self, cal_function):
        sum_mapk = 0.0
        for k, user in enumerate(self.users):
            user_id = k
            user_coupons = self.train_coupon_vec[user["coupon_ids"]]
            maxmin_columns = self.__maxmin_columns(user["coupon_ids"])
            hist_feat = self.__purchase_history_features(user_coupons, maxmin_columns)
            feats = np.empty((len(self.valid_coupon_vec),
                              len(user["user"]) + len(hist_feat) + len(self.valid_coupon_vec[0])),
                             dtype=np.float32)
            coupon_ids = []
            for i in xrange(len(self.valid_coupon_vec)):
                coupon_id = i
                feats[i][:] = np.hstack((user["user"], hist_feat, self.valid_coupon_vec[i]))
                coupon_ids.append(coupon_id)

            sum_mapk += cal_function({"user_id": user_id, "coupon_ids": coupon_ids, "coupon_feats": feats})
            if (k % 100 == 0):
                print ("each valid .. %d/%d\r" % (k, len(self.users))),
                sys.stdout.flush()
        print "\n"
        return sum_mapk



