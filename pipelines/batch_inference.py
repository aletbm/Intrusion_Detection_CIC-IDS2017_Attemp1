import pandas as pd
import sys
import os
from prefect import flow, task
from google.cloud import storage
import cloudpickle

RUN_ID = "14443851117c4ad2a1e39a3182b4c10e"
artifacts_path = f"./models/1/{RUN_ID}/artifacts/"

model_path = f"{artifacts_path}/xgboost_model/model.pkl"
scaler_path = f"{artifacts_path}/preprocessing/scaler.pkl"
le_path = f"{artifacts_path}/preprocessing/le.pkl"

with open(model_path, "rb") as f:
    model = cloudpickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = cloudpickle.load(f)

with open(le_path, "rb") as f:
    le = cloudpickle.load(f)

@task
def load_data(filename, columns):
    df = pd.read_parquet(filename)
    return df[columns]

def split_X_y(df, target):
    return df.drop(target, axis=1), df[target]

@task
def prepare_data(test):
    X_test, y_test = split_X_y(test, "label")
    X_test = scaler.transform(X_test)
    y_test = le.transform(y_test)
    return X_test, y_test

@task
def apply_model(X_test):
    y_pred = model.predict(X_test)
    return y_pred

@task
def make_result(df, y_pred):
    df['prediction'] = y_pred
    df_result = df[["prediction"]].copy()
    return df_result

@task
def save_result(df_result, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = f"{output_folder}/predictions.parquet"
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return

def upload_blob(project_id, bucket_name, source_file_name, destination_blob_name):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded as {destination_blob_name}")
    return

@task
def upload2cloud(project_id, bucket_name, output_folder):
    filename = f"{output_folder}/predictions.parquet"
    upload_blob(
        project_id=project_id,
        bucket_name=bucket_name,
        source_file_name=filename,
        destination_blob_name=filename
    )
    return

@flow(name="Intrusion Inference Pipeline", retries=1, retry_delay_seconds=300)
def intrusion_inference_pipeline(project_id, bucket_name, filepath):
    output_folder = f"output"
    columns = ['flow_iat_mean', 
               'psh_flag_count', 
               'fwd_packets/s', 
               'bwd_packet_length_std', 
               'init_win_bytes_forward', 
               'active_min', 
               'bwd_packets/s', 
               'subflow_fwd_bytes', 
               'active_std', 
               'urg_flag_count', 
               'init_win_bytes_backward', 
               'act_data_pkt_fwd', 
               'fwd_iat_std', 
               'bwd_packet_length_min', 
               'fwd_iat_total', 
               'min_packet_length', 
               'total_fwd_packets', 
               'fwd_packet_length_mean', 
               'fwd_packet_length_std', 
               'fin_flag_count', 
               'bwd_iat_std', 
               'min_seg_size_forward', 
               'bwd_iat_max', 
               'label']

    df = load_data(filepath, columns)
    X_test, _ = prepare_data(df)
    y_pred = apply_model(X_test)
    df_result = make_result(df, y_pred)
    save_result(df_result, output_folder)
    upload2cloud(project_id, bucket_name, output_folder)
    return

if __name__ == '__main__':
    filepath = sys.argv[1]
    project_id="plucky-haven-463121-j1"
    bucket_name='plucky-haven-463121-j1-predictions'
    intrusion_inference_pipeline(project_id, bucket_name, filepath)