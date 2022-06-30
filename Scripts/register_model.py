import argparse
import os
import pickle

import mlflow
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import xgboost as xgb
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "xgboost-hyperopt"
EXPERIMENT_NAME = "xgboost-best-models"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.xgboost.autolog()

SPACE = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train1.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid1.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test1.pkl"))

    with mlflow.start_run():
        params = space_eval(SPACE, params)
        rf = xgb.Regressor(**params)
        rf.fit(X_train, y_train)

        # evaluate model on the validation and test sets
        valid_rmse = mean_squared_error(y_valid, rf.predict(X_valid), squared=False)
        mlflow.log_metric("valid_rmse", valid_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)


def run(data_path, log_top):

    client = MlflowClient()

    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    # best_run = client.search_runs( ...  )[0]

    # register the best model
    # mlflow.register_model( ... )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/home/david/Elvis/macro-eyes/ML_Assignment/Data/",
        help="the location where the prepared data was saved."
    )
    parser.add_argument(
        "--top_n",
        default=5,
        type=int,
        help="the top 'top_n' models will be evaluated to decide which model to promote."
    )
    args = parser.parse_args()

    run(args.data_path, args.top_n)