from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
import joblib
import os
from config import Config
from app.services.preprocessor import preprocessor
from app.utils.logger import logger

class FeatureEngineer:
    def __init__(self):
        self.vectorizer = None
    
    def create_vectorizer(self, tfidf_config: dict = None) -> TfidfVectorizer:
        config = Config.TFIDF_DEFAULT_CONFIG.copy()
        if tfidf_config:
            config.update(tfidf_config)
        
        logger.info(f"Creating TF-IDF vectorizer with config: {config}")
        
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            ngram_range=config["ngram_range"],
            min_df=config["min_df"],
            max_df=config["max_df"],
            max_features=config["max_features"],
            sublinear_tf=config["sublinear_tf"]
        )
        return self.vectorizer
    
    def fit_transform(self, texts, tfidf_config: dict = None):
        processed_texts = preprocessor.tokenize_batch(texts)
        logger.info(f"Fitting TF-IDF on {len(processed_texts)} documents")
        
        vectorizer = self.create_vectorizer(tfidf_config)
        X = vectorizer.fit_transform(processed_texts)
        
        logger.info(f"TF-IDF matrix shape: {X.shape}, sparse: {isinstance(X, type(vectorizer.transform(['test'])))}")
        return X, vectorizer
    
    def transform(self, texts, vectorizer: TfidfVectorizer = None):
        vec = vectorizer or self.vectorizer
        if vec is None:
            raise ValueError("No vectorizer available. Call fit_transform first or provide one.")
        
        processed_texts = preprocessor.tokenize_batch(texts)
        X = vec.transform(processed_texts)
        return X
    
    def save_vectorizer(self, vectorizer: TfidfVectorizer, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(vectorizer, filepath)
        logger.info(f"Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath: str) -> TfidfVectorizer:
        self.vectorizer = joblib.load(filepath)
        logger.info(f"Vectorizer loaded from {filepath}")
        return self.vectorizer

feature_engineer = FeatureEngineer()
