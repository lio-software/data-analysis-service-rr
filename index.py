import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class SentimentRequest(BaseModel):
    text: str

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
