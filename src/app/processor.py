"""Processor module for sentiment analysis of earnings-related messages using FinBERT with VADER fallback."""

import logging
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Initialize VADER once
_vader = SentimentIntensityAnalyzer()

# Load FinBERT model and tokenizer (cached after first load)
try:
    _finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    _finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    _finbert_model.eval()
    _finbert_device = "cuda" if torch.cuda.is_available() else "cpu"
    _finbert_model.to(_finbert_device)
    _labels = ["positive", "negative", "neutral"]
    _finbert_available = True
except Exception as e:
    logger.warning(f"FinBERT unavailable. Falling back to VADER only. Error: {e}")
    _finbert_available = False


def analyze_sentiment(
    text: str, backend: Optional[Literal["finbert", "vader", "auto"]] = "auto"
) -> dict:
    """
    Analyze the sentiment of an earnings-related message using FinBERT or VADER.

    Args:
        text (str): The message to analyze.
        backend (str): 'finbert', 'vader', or 'auto' (default = 'auto').

    Returns:
        dict: {
            "original_text": str,
            "sentiment": str,
            "confidence": float,
            "probabilities": dict[str, float],
            "engine": str
        }
    """
    if not text or not text.strip():
        return {
            "original_text": text,
            "sentiment": "unknown",
            "confidence": None,
            "probabilities": None,
            "engine": None,
        }

    # Try FinBERT first (if available)
    if backend == "finbert" or (backend == "auto" and _finbert_available):
        try:
            inputs = _finbert_tokenizer(text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(_finbert_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = _finbert_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)[0]

            sentiment_idx = torch.argmax(probs).item()
            sentiment = _labels[sentiment_idx]
            confidence = round(probs[sentiment_idx].item(), 4)

            return {
                "original_text": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": {
                    label: round(probs[i].item(), 4)
                    for i, label in enumerate(_labels)
                },
                "engine": "finbert",
            }
        except Exception as e:
            logger.error(f"FinBERT failed: {e}. Falling back to VADER.")

    # Fallback to VADER
    try:
        scores = _vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "original_text": text,
            "sentiment": sentiment,
            "confidence": round(abs(compound), 4),
            "probabilities": {
                "positive": round(scores["pos"], 4),
                "neutral": round(scores["neu"], 4),
                "negative": round(scores["neg"], 4),
            },
            "engine": "vader",
        }
    except Exception as e:
        logger.error(f"VADER sentiment analysis also failed: {e}")
        return {
            "original_text": text,
            "sentiment": "error",
            "confidence": None,
            "probabilities": None,
            "engine": "vader",
        }
