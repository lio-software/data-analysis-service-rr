FROM python:3.8-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn xgboost mysql-connector-python

EXPOSE 3005

CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "3005"]
