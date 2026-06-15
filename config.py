import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    PORT = 5001
    HOST = "0.0.0.0"
    
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    DB_PATH = os.path.join(DATA_DIR, "ml_toy.db")
    
    STOPWORDS_PATH = os.path.join(DATA_DIR, "stopwords.txt")
    
    MAX_SNAPSHOTS = 5
    
    TRAINING_THREAD_POOL_SIZE = 1
    
    TFIDF_DEFAULT_CONFIG = {
        "ngram_range": (1, 1),
        "min_df": 1,
        "max_df": 1.0,
        "max_features": None,
        "sublinear_tf": True
    }
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    TOP_N_PREDICTIONS = 3

for _dir in [Config.DATA_DIR, Config.MODEL_DIR, Config.LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)
