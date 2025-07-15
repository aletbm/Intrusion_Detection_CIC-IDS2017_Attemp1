import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task
import os

seed = 42
input_folder = "./"
data_folder = input_folder + "data/"
model_folder = input_folder + "models/"


@task
def load_data():
    train = pd.read_parquet(data_folder + "train.parquet")
    val = pd.read_parquet(data_folder + "val.parquet")
    test = pd.read_parquet(data_folder + "test.parquet")
    return train, val, test


def data_scaling(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return scaler, X_train, X_val, X_test


def label_encoding(y_train, y_val, y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)
    return le, y_train, y_val, y_test


def split_X_y(df, target):
    return df.drop(target, axis=1), df[target]


@task
def prepare_data(train, val, test):
    target = "label"
    X_train, y_train = split_X_y(train, target)
    X_val, y_val = split_X_y(val, target)
    X_test, y_test = split_X_y(test, target)
    scaler, X_train, X_val, X_test = data_scaling(X_train, X_val, X_test)
    le, y_train, y_val, y_test = label_encoding(y_train, y_val, y_test)

    with open(model_folder + "scaler.pkl", "wb") as f:
        cloudpickle.dump(scaler, f)

    with open(model_folder + "le.pkl", "wb") as f:
        cloudpickle.dump(le, f)

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_scores(y_true, y_pred, y_pred_proba):
    return {
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "ROC AUC": roc_auc_score(
            y_true, y_pred_proba, average="weighted", multi_class="ovr"
        ),
    }


@task
def training(X_train, y_train, X_val, y_val, X_test, y_test):
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("intrusion_detection_experiment")

    with mlflow.start_run():
        xgb_clf = XGBClassifier(
            n_estimators=1000,
            eta=0.1,
            objective="multi:softmax",
            eval_metric="auc",
            early_stopping_rounds=10,
            class_weight=dict(zip(np.unique(y_train), class_weights)),
            verbosity=1,
            n_jobs=-1,
            seed=seed,
        )
        xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        y_pred = xgb_clf.predict(X_test)
        y_pred_proba = xgb_clf.predict_proba(X_test)
        scores = get_scores(y_test, y_pred, y_pred_proba)

        mlflow.log_param("feature_importances_", xgb_clf.feature_importances_)
        mlflow.log_param("best_score", xgb_clf.best_score)
        mlflow.log_param("best_iteration", xgb_clf.best_iteration)
        mlflow.log_param("params", xgb_clf.get_params(deep=True))
        mlflow.log_param("intercept_", xgb_clf.intercept_)
        mlflow.log_metric("Balanced Accuracy", scores["Balanced Accuracy"])
        mlflow.log_metric("F1 Score", scores["F1 Score"])
        mlflow.log_metric("Precision", scores["Precision"])
        mlflow.log_metric("Recall", scores["Recall"])
        mlflow.log_metric("ROC AUC", scores["ROC AUC"])

        model_info = mlflow.sklearn.log_model(
            sk_model=xgb_clf,
            artifact_path="xgboost_model",
            registered_model_name="MyXGBClassifier",
            signature=mlflow.models.infer_signature(X_train, xgb_clf.predict(X_train)),
            input_example=X_train[0:2],
        )
        scaler_path = model_folder + "scaler.pkl"
        le_path = model_folder + "le.pkl"

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/xgboost_model"

        model_info = mlflow.register_model(model_uri, "MyXGBClassifier")

        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
        mlflow.log_artifact(le_path, artifact_path="preprocessing")

        os.remove(scaler_path)
        os.remove(le_path)

        client = MlflowClient()

        client.transition_model_version_stage(
            name="MyXGBClassifier",
            version=model_info.version,
            stage="Staging",
            archive_existing_versions=False,
        )

    return xgb_clf


@flow(name="Intrusion ML Pipeline", retries=1, retry_delay_seconds=300)
def intrusion_pipeline():
    train, val, test = load_data()
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = prepare_data(train, val, test)
    training(X_train, y_train, X_val, y_val, X_test, y_test)
    return


if __name__ == "__main__":
    intrusion_pipeline()

# prefect server start
# mlflow server --backend-store-uri sqlite:///mlflow.db \
# --default-artifact-root ./models
