from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI()

# Allow all CORS (for frontend to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained sentiment analysis model
classifier = pipeline("sentiment-analysis")

@app.get("/")
def read_root():
    return {"message": "Sentiment API is running."}

@app.post("/analyze")
async def analyze_sentiment(request: Request):
    data = await request.json()
    text = data.get("text")
    result = classifier(text)[0]
    return {
        "label": result["label"],
        "score": round(result["score"], 4)
    }
