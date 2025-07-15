from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
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

app = FastAPI()


class InputData(BaseModel):
    features: list


@app.post("/predict")
def predict(data: InputData):
    array = np.array(data.features).reshape(1, -1)
    array = scaler.transform(array)
    prediction = model.predict(array)
    return {"prediction": le.inverse_transform(prediction)[0]}
