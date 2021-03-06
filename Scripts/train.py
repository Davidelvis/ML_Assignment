import argparse
import os
import pickle

import xgboost as xgb
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):

    X_train, y_train = load_pickle(os.path.join(data_path, "train1.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid1.pkl"))

    rf = xgb.XGBRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_valid)

    rmse = mean_squared_error(y_valid, y_pred, squared=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/home/david/Elvis/macro-eyes/ML_Assignment/Data/",
        help="the location where the prepared data was saved."
    )
    args = parser.parse_args()

    run(args.data_path)