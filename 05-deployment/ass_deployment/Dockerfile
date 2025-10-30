FROM agrigorev/zoomcamp-model:2025

RUN pip install scikit-learn==1.6.1
RUN pip install uvicorn
RUN pip install fastapi

COPY "predict_web_service_Q4.py" ./

ENTRYPOINT ["uvicorn", "predict_web_service_Q4:app", "--host", "0.0.0.0", "--port", "9696"]
