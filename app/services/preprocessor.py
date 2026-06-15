import jieba
from config import Config
from app.utils.logger import logger

class TextPreprocessor:
    def __init__(self):
        self.stopwords = self._load_stopwords()
        logger.info(f"Loaded {len(self.stopwords)} stopwords")
    
    def _load_stopwords(self):
        stopwords = set()
        try:
            with open(Config.STOPWORDS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
        except FileNotFoundError:
            logger.warning(f"Stopwords file not found at {Config.STOPWORDS_PATH}, using empty set")
        return stopwords
    
    def tokenize(self, text):
        if not text or not isinstance(text, str):
            return []
        
        words = jieba.lcut(text)
        filtered_words = []
        
        for word in words:
            word = word.strip()
            if word and word not in self.stopwords and len(word) > 0:
                filtered_words.append(word)
        
        return filtered_words
    
    def tokenize_for_tfidf(self, text):
        tokens = self.tokenize(text)
        return " ".join(tokens)
    
    def tokenize_batch(self, texts):
        return [self.tokenize_for_tfidf(text) for text in texts]

preprocessor = TextPreprocessor()
