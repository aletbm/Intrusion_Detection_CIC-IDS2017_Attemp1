import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipelines import training_flow
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

input_folder = "./"
data_folder = input_folder + "data/"


def test_data_scaling():
    train = pd.read_parquet(data_folder + "train.parquet")
    val = pd.read_parquet(data_folder + "val.parquet")
    test = pd.read_parquet(data_folder + "test.parquet")

    X_train, _ = train.drop("label", axis=1), train["label"]
    X_val, _ = val.drop("label", axis=1), val["label"]
    X_test, _ = test.drop("label", axis=1), test["label"]

    _, act_X_train, act_X_val, act_X_test = training_flow.data_scaling(
        X_train, X_val, X_test
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    assert np.std(X_train) == np.std(act_X_train)
    assert np.std(X_val) == np.std(act_X_val)
    assert np.std(X_test) == np.std(act_X_test)

    assert len(X_train) == len(act_X_train)
    assert len(X_val) == len(act_X_val)
    assert len(X_test) == len(act_X_test)


def test_label_encoding():
    train = pd.read_parquet(data_folder + "train.parquet")
    val = pd.read_parquet(data_folder + "val.parquet")
    test = pd.read_parquet(data_folder + "test.parquet")

    _, y_train = train.drop("label", axis=1), train["label"]
    _, y_val = val.drop("label", axis=1), val["label"]
    _, y_test = test.drop("label", axis=1), test["label"]

    _, act_y_train, act_y_val, act_y_test = training_flow.label_encoding(
        y_train, y_val, y_test
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.transform(y_val)
    y_test = le.transform(y_test)

    assert np.array_equal(np.unique(y_train), np.unique(act_y_train))
    assert np.array_equal(np.unique(y_val), np.unique(act_y_val))
    assert np.array_equal(np.unique(y_test), np.unique(act_y_test))

    assert len(y_train) == len(act_y_train)
    assert len(y_val) == len(act_y_val)
    assert len(y_test) == len(act_y_test)
