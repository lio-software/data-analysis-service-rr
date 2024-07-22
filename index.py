import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import mysql.connector
from xgboost import XGBRegressor

class SentimentRequest(BaseModel):
    text: str

class TimeDF(BaseModel):
    uuid: str

data = pd.read_csv("dataset.csv")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vect, y_train)


@app.post("/api/v1/sentiment")
async def analize_sentiment(request: SentimentRequest = Body(...)):
    messages = [request.text]
    text_vect = vectorizer.transform(messages)
    predictions = model.predict(text_vect)
    result = predictions[0]
    if result == 0:
        return {"data": {"sentiment": 0}}
    return {"data": {"sentiment": 1}}


@app.post("/api/v1/sentiment/time_df")
async def get_time_df(request: TimeDF = Body(...)):
    uuid = request.uuid
    mydb = mysql.connector.connect(
        host=os.getenv("DB_URL"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE")
    )
    cursor = mydb.cursor()
    cursor.execute(f"SELECT * FROM rentals WHERE lessor_id = '{uuid}' AND status = 'FINISHED';")
    result = cursor.fetchall()
    cursor.close()
    mydb.close()

    data = pd.DataFrame(result, columns=['id', 'lesse_id', 'lessor_id', 'vehicle_id', 'start_date', 'end_date',
       'uuid', 'total_amount', 'status', 'createdAt', 'updatedAt'])

    df = data[['end_date']].copy()
    df['end_date'] = pd.to_datetime(df['end_date'])
    df.sort_values('end_date', inplace=True)
    df = df[(df['end_date'] >= '2024-01-01') & (df['end_date'] <= '2024-05-31')]

    df['RENTALS'] = 3
    df = df.groupby('end_date').size().reset_index(name='RENTALS')
    df.set_index('end_date', inplace=True)

    def create_attributes(dfe):
        dfe = dfe.copy()
        dfe['day'] = dfe.index.day
        dfe['dayofweek'] = dfe.index.dayofweek
        dfe['month'] = dfe.index.month
        dfe['quarter'] = dfe.index.quarter
        dfe['year'] = dfe.index.year
        dfe['dayofyear'] = dfe.index.dayofyear
        return dfe

    new_df = create_attributes(df)

    features = ['day','dayofweek','month','quarter','year','dayofyear']
    target = ['RENTALS']

    X_full = new_df[features]
    y_full = new_df[target]

    xgb_regf = XGBRegressor(booster='gbtree',
                                seed=42,
                                n_estimators=1000,
                                early_stopping_rounds=50,
                                objective='reg:squarederror',
                                reg_lambda=0.001,
                                max_depth=5,
                                eta=0.01)
    xgb_regf.fit(X_full, y_full,
            eval_set=[(X_full, y_full)],
            verbose=100)

    pred_dates = pd.date_range('2024-06-01','2024-07-01', freq='D')
    preds_df = pd.DataFrame(index=pred_dates)

    preds_df['Future'] = True
    new_df['Future'] = False

    pred_pw = pd.concat([new_df, preds_df])

    pred_pw = pred_pw.copy()
    pred_pw = create_attributes(pred_pw)

    future_pred_pw = pred_pw.query('Future').copy()

    future_pred_pw['RENTALS'] = xgb_regf.predict(future_pred_pw[features])

    future_pred_pw['RENTALS'] = future_pred_pw['RENTALS'].astype(int)

    return {"real_df": new_df['RENTALS'].to_dict(), "prediction_df": future_pred_pw['RENTALS'].to_dict()}
