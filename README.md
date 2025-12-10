# Fake News Detection System

A machine learning-based system to detect fake news articles using TF-IDF and Transformer (DistilBERT) models.

## Project Structure

```
Project170/
├── data/                          # Dataset files (not included - see Data Setup)
│   ├── Fake.csv                   # Fake news dataset
│   ├── True.csv                   # Real news dataset
│   └── merged_clean.csv           # Preprocessed merged dataset
├── model/                         # Trained model files (not included - see Training)
│   ├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
│   ├── logistic_model.pkl         # Baseline logistic regression model
│   └── distilbert_classifier.pkl  # Transformer-based classifier
├── src/
│   ├── prepare_dataset.py         # Data preprocessing script
│   ├── baseline_model.py          # Train TF-IDF + Logistic Regression
│   ├── train_transformer.py       # Train DistilBERT + Logistic Regression
│   ├── predict.py                 # Baseline prediction pipeline
│   ├── predict_transformer.py     # Transformer prediction pipeline
│   ├── test_headline.py           # CLI tool for baseline model testing
│   └── test_headline_transformer.py # CLI tool for transformer testing
├── results/                       # Generated plots and explanations
├── explain_tfidf.py              # Global explainability (top words)
├── explain_local.py              # Local per-text word importance
├── lime_explain.py               # LIME interactive explanations
├── app.py                        # FastAPI backend (optional)
└── README.md
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```

### 2. Create Virtual Environment
```bash
python3 -m venv env
source env/bin/activate  # On macOS/Linux
# or
env\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn torch transformers lime joblib
# Optional for API:
pip install fastapi uvicorn
```

### 4. Data Setup
Download the Kaggle Fake News dataset:
- [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Place `Fake.csv` and `True.csv` in the `data/` folder

### 5. Prepare Dataset
```bash
cd src
python prepare_dataset.py
```
This creates `merged_clean.csv` with ~44,898 samples.

## Training Models

### Baseline Model (TF-IDF + Logistic Regression)
```bash
cd src
python baseline_model.py
```
- Accuracy: ~98.8%
- Saves to `model/tfidf_vectorizer.pkl` and `model/logistic_model.pkl`

### Transformer Model (DistilBERT + Logistic Regression)
```bash
cd src
python train_transformer.py
```
- Accuracy: ~99.1%
- Saves to `model/distilbert_classifier.pkl`

## Usage

### Test Headlines (Terminal Demo)

**Baseline Model:**
```bash
cd src
python test_headline.py "Your news headline here"
```

**Transformer Model:**
```bash
cd src
python test_headline_transformer.py "Your news headline here"
```

### Example Output
```
============================================================
                FAKE NEWS DETECTION RESULT
============================================================
Credibility Score: 0.87
Label: Likely Real
Confidence: High
============================================================
```

### Prediction Output Format
- **Credibility Score**: 0.0 (fake) to 1.0 (real)
- **Label**: 
  - `Likely Real` (score ≥ 0.75)
  - `Uncertain` (0.40 ≤ score < 0.75)
  - `Likely Fake` (score < 0.40)
- **Confidence**: High/Medium/Low

## Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| TF-IDF + Logistic Regression | 98.80% | 98.74% |
| DistilBERT + Logistic Regression | 99.10% | 99.06% |

## Explainability Features (For Report)

### Global Explainability
```bash
python explain_tfidf.py
```
Shows top words associated with real/fake news.

### Local Explainability
```bash
python explain_local.py
```
Shows word importance for specific texts.

### LIME Explanations
```bash
python lime_explain.py
```
Generates interactive HTML explanations.

## API (Optional)

Start the FastAPI server:
```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /predict` - Get prediction for text
- `GET /health` - Health check

## Team Members
- [Add team member names]

## Dataset Source
- Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Contains US political news from 2017-2018

## Notes
- Model performs best on political news content
- News with source attribution (e.g., "Reuters") tends to score as more credible
- Sensational headlines without sources may score as potentially fake
