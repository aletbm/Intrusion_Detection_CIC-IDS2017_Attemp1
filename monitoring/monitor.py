import pandas as pd
import cloudpickle
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from evidently import Report, Dataset, DataDefinition
from evidently.presets import (
    DataDriftPreset,
    ClassificationPreset,
    DataSummaryPreset,
)
from evidently import MulticlassClassification
from prefect import flow, task
import os
import requests
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipelines.training_flow import intrusion_pipeline

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "MyXGBClassifier"
MODEL_STAGE = "Staging"
ARTIFACT_DIR = "monitoring/artifacts"
DRIFT_THRESHOLD = -1
SLACK_WEBHOOK_URL = (
    "https://hooks.slack.com/services/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"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


@task
def load_model_and_artifacts():
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])[0]
    run_id = latest.run_id

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    scaler_path = client.download_artifacts(
        run_id, "preprocessing/scaler.pkl", ARTIFACT_DIR
    )
    le_path = client.download_artifacts(run_id, "preprocessing/le.pkl", ARTIFACT_DIR)

    with open(scaler_path, "rb") as f:
        scaler = cloudpickle.load(f)

    with open(le_path, "rb") as f:
        le = cloudpickle.load(f)

    return model, scaler, le


@task
def load_data():
    train_df = pd.read_parquet("data/train.parquet")
    test_df = pd.read_parquet("data/test.parquet")
    return train_df, test_df


@task
def prepare_datasets(model, scaler, le, train_df, test_df, target_col="label"):
    feature_cols = train_df.drop(columns=[target_col]).columns.tolist()

    def make_dataset(df):
        X = scaler.transform(df[feature_cols])
        y = le.transform(df[target_col])
        preds = model.predict(X)
        df2 = pd.DataFrame(X, columns=feature_cols)
        df2["target"] = y
        df2["prediction"] = preds

        return Dataset.from_pandas(
            df2,
            data_definition=DataDefinition(
                classification=[
                    MulticlassClassification(
                        target="target", prediction_labels="prediction"
                    )
                ],
                numerical_columns=feature_cols,
            ),
        )

    ds_train = make_dataset(train_df)
    ds_test = make_dataset(test_df)
    return ds_train, ds_test


@task
def run_monitoring(ds_train, ds_test):
    report = Report(
        metrics=[DataDriftPreset(), DataSummaryPreset(), ClassificationPreset()]
    )
    result = report.run(reference_data=ds_train, current_data=ds_test)
    result.save_html("monitoring/full_monitor_report.html")

    report_dict = result.json()
    report_dict = json.loads(report_dict)

    for metric in report_dict["metrics"]:
        metric_id = metric["metric_id"]
        value = metric["value"]

        if "DriftedColumnsCount" in metric_id:
            drift_score = value["share"]
            break

    print(f"Drift score: {drift_score}")
    return drift_score


@task
def send_slack_alert(message: str):
    if not SLACK_WEBHOOK_URL:
        print("Slack webhook not configured.")
        return
    payload = {"text": message}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)
    if response.status_code == 200:
        print("Slack alert sent successfully.")
    else:
        print(f"Failed to send Slack alert: {response.status_code} {response.text}")


@flow(name="Retrain Model Flow")
def retrain_flow():
    print("Starting model retraining...")
    intrusion_pipeline()
    print("Retraining completed.")


@task
def check_drift_and_maybe_retrain(drift_score: float):
    if drift_score > DRIFT_THRESHOLD:
        alert_msg = f"ALERT: Drift detected (score={drift_score:.3f}) > threshold ({DRIFT_THRESHOLD})"
        print(alert_msg)
        send_slack_alert(alert_msg)
        retrain_flow.submit()  # Async retraining
        return True
    else:
        print(f"Drift is acceptable ({drift_score:.3f} ≤ {DRIFT_THRESHOLD})")
        return False


@flow(name="Monitoring + Conditional Retraining")
def monitoring_flow():
    model, scaler, le = load_model_and_artifacts()
    train_df, test_df = load_data()
    ds_train, ds_test = prepare_datasets(model, scaler, le, train_df, test_df)
    drift_score = run_monitoring(ds_train, ds_test)
    check_drift_and_maybe_retrain(drift_score)


if __name__ == "__main__":
    send_slack_alert("Monitoring started.")
    monitoring_flow()

# prefect server start
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models
