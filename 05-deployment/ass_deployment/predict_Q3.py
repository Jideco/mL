import pickle
import os

print('importing model...')

os.system("wget https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/pipeline_v1.bin")

print('...')
print('Finished importing model...')
print('...')

model_name = 'pipeline_v1.bin'

with open(model_name, 'rb') as f_in:
    pipeline = pickle.load(f_in)


data_point = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

churn = pipeline.predict_proba(data_point)[0,1]
print('prob of churning =', churn)