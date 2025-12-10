# app.py
# FastAPI Backend for Fake News Detection
# Endpoints:
#   POST /predict → returns score, label, confidence
#   POST /explain → returns TF-IDF word importance (optional)
#   GET  /health  → health check

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import os
import numpy as np

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")

# Credibility thresholds
THRESHOLD_REAL = 0.75
THRESHOLD_UNCERTAIN = 0.40

# Text limits
MAX_TEXT_LENGTH = 50000
MIN_TEXT_LENGTH = 10

# ---------------------------
# LOAD MODEL (once at startup)
# ---------------------------
print("[INFO] Loading model and vectorizer...")
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("[INFO] Model loaded successfully!")

# Get feature names and coefficients for explanations
feature_names = np.array(vectorizer.get_feature_names_out())
coef = model.coef_[0]

# ---------------------------
# FASTAPI APP
# ---------------------------
app = FastAPI(
    title="Fake News Detection API",
    description="Analyze news articles and get credibility scores",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# REQUEST/RESPONSE MODELS
# ---------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=MIN_TEXT_LENGTH, max_length=MAX_TEXT_LENGTH,
                      description="News headline or article text to analyze")

class PredictResponse(BaseModel):
    score: float = Field(..., description="Credibility score (0-1, higher = more likely real)")
    label: str = Field(..., description="Likely Real / Uncertain / Likely Fake")
    confidence: str = Field(..., description="Low / Medium / High")

class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=MIN_TEXT_LENGTH, max_length=MAX_TEXT_LENGTH)
    top_k: int = Field(default=10, ge=1, le=50, description="Number of top words to return")

class WordImportance(BaseModel):
    token: str
    weight: float

class ExplainResponse(BaseModel):
    score: float
    label: str
    confidence: str
    word_importance: list[WordImportance]

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def get_credibility_score(text: str) -> float:
    """Get probability of being REAL (0-1)."""
    text_vec = vectorizer.transform([text])
    prob_real = model.predict_proba(text_vec)[0, 1]
    return float(prob_real)

def get_label(score: float) -> str:
    """Convert score to label."""
    if score >= THRESHOLD_REAL:
        return "Likely Real"
    elif score >= THRESHOLD_UNCERTAIN:
        return "Uncertain"
    else:
        return "Likely Fake"

def get_confidence(score: float) -> str:
    """Get confidence level based on score."""
    if score >= 0.90 or score <= 0.10:
        return "High"
    elif score >= 0.75 or score <= 0.25:
        return "Medium"
    else:
        return "Low"

def get_word_importance(text: str, top_k: int = 10) -> list[dict]:
    """Get word-level importance for explanation."""
    tfidf_vec = vectorizer.transform([text])
    tfidf_array = tfidf_vec.toarray()[0]
    word_importance = tfidf_array * coef
    
    # Get non-zero indices
    nonzero_idx = np.where(tfidf_array > 0)[0]
    
    # Build importance list
    importance_list = []
    for idx in nonzero_idx:
        importance_list.append({
            "token": feature_names[idx],
            "weight": round(float(word_importance[idx]), 4)
        })
    
    # Sort by absolute importance
    importance_list.sort(key=lambda x: abs(x["weight"]), reverse=True)
    
    return importance_list[:top_k]

# ---------------------------
# API ENDPOINTS
# ---------------------------
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Fake News Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Get credibility score and label",
            "/explain": "POST - Get word-level explanation",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Analyze news text and return credibility prediction.
    
    - **text**: News headline or article to analyze (10-50000 characters)
    
    Returns:
    - **score**: Credibility score (0-1, higher = more likely real)
    - **label**: Likely Real / Uncertain / Likely Fake
    - **confidence**: Low / Medium / High
    """
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        score = get_credibility_score(text)
        label = get_label(score)
        confidence = get_confidence(score)
        
        return PredictResponse(
            score=round(score, 4),
            label=label,
            confidence=confidence
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Get word-level explanation for a prediction.
    
    - **text**: News headline or article to analyze
    - **top_k**: Number of most important words to return (default: 10)
    
    Returns prediction plus word importance scores.
    Positive weights push toward REAL, negative toward FAKE.
    """
    try:
        text = request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        score = get_credibility_score(text)
        label = get_label(score)
        confidence = get_confidence(score)
        word_importance = get_word_importance(text, request.top_k)
        
        return ExplainResponse(
            score=round(score, 4),
            label=label,
            confidence=confidence,
            word_importance=[WordImportance(**w) for w in word_importance]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

# ---------------------------
# RUN SERVER
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
