import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any


app = FastAPI(title="client-churn-prediction")

model_name = 'pipeline_v1.bin'

with open(model_name, 'rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)