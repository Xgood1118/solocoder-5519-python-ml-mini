from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict
from config import Config
from app.services.feature_engineer import feature_engineer
from app.utils.logger import logger

class Predictor:
    def __init__(self):
        self.model: MultinomialNB = None
        self.vectorizer: TfidfVectorizer = None
        self.classes_: List[str] = None
        self.snapshot_id: int = None
    
    def load_model(self, model: MultinomialNB, vectorizer: TfidfVectorizer, classes: List[str], snapshot_id: int = None):
        self.model = model
        self.vectorizer = vectorizer
        self.classes_ = classes
        self.snapshot_id = snapshot_id
        logger.info(f"Predictor loaded with {len(classes)} classes, snapshot_id: {snapshot_id}")
    
    def predict_single(self, text: str, top_n: int = None) -> Dict:
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded. Call load_model first.")
        
        top_n = top_n or Config.TOP_N_PREDICTIONS
        X = feature_engineer.transform([text], self.vectorizer)
        
        proba = self.model.predict_proba(X)[0]
        top_indices = np.argsort(proba)[::-1][:top_n]
        
        predictions = []
        for idx in top_indices:
            predictions.append({
                "label": self.classes_[idx],
                "confidence": float(proba[idx])
            })
        
        predicted_label = self.classes_[top_indices[0]]
        confidence = float(proba[top_indices[0]])
        
        result = {
            "text": text,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "top_predictions": predictions,
            "snapshot_id": self.snapshot_id
        }
        
        logger.debug(f"Predicted '{text}' -> {predicted_label} (confidence: {confidence:.4f})")
        return result
    
    def predict_batch(self, texts: List[str], top_n: int = None) -> List[Dict]:
        results = []
        for text in texts:
            results.append(self.predict_single(text, top_n))
        logger.info(f"Batch prediction completed for {len(texts)} texts")
        return results
    
    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

predictor = Predictor()
