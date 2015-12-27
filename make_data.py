import numpy as np
import datetime
import argparse
from dataset import Dataset

parser = argparse.ArgumentParser(description='coupon-purchase-prediction-solution')
parser.add_argument('--validation', '-v', action="store_true",
                    help='Validation mode')
args = parser.parse_args()

dataset = Dataset(datadir="../data")
if args.validation:
    valid_days = datetime.timedelta(days=7*4)
    dataset.load(validation_timedelta=valid_days)
    dataset.save_pkl("result/valid_28.pkl")
else:
    dataset.load()
    dataset.save_pkl("result/all_data.pkl")
