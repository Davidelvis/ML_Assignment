import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    df = pd.read_pickle(filename)

    categorical = ['dob','gender','region','district','im_date','successful','reason_unsuccesful']
    df[categorical] = df[categorical].astype(str)

    return df


def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    categorical = ['dob','gender','region','district','im_date','successful','reason_unsuccesful']
    numerical = ['fac_id','pat_id','DTP','OPV','lat','long']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv


def run(raw_data_path: str, dest_path: str, dataset: str = "immun"):
    # load pickle files
    df_train = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_train.pkl")
    )
    df_valid = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_valid.pkl")
    )
    df_test = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_test.pkl")
    )

    # extract the target
    target = 'full_dose'
    y_train = df_train[target].values
    y_valid = df_valid[target].values
    y_test = df_test[target].values

    # fit the dictvectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_valid, _ = preprocess(df_valid, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    # create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # save dictvectorizer and datasets
    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train1.pkl"))
    dump_pickle((X_valid, y_valid), os.path.join(dest_path, "valid1.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test1.pkl"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        default="/home/david/Elvis/macro-eyes/ML_Assignment/Data/",
        help="the location where the raw data was saved"
    )
    parser.add_argument(
        "--dest_path",
        default="/home/david/Elvis/macro-eyes/ML_Assignment/Data/",
        help="the location where the resulting files will be saved."
    )
    args = parser.parse_args()

    run(args.raw_data_path, args.dest_path)