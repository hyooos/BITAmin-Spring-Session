from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()
classifier = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message":"API server is running"}

@app.post("/predict")
def predict_sentiment(request: TextInput):
    result = classifier(request.text)
    return {"prediction":result}

