import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import joblib

from config import Config
from app.models import Category, ModelSnapshot, get_db_session
from app.services.data_loader import data_loader
from app.services.feature_engineer import feature_engineer
from app.services.evaluator import evaluator
from app.services.predictor import predictor
from app.utils.logger import logger

class TrainingStatus:
    def __init__(self):
        self._status = {}
        self._lock = threading.Lock()
    
    def init_training(self, training_id: str):
        with self._lock:
            self._status[training_id] = {
                "status": "running",
                "progress": 0,
                "message": "Training started",
                "created_at": datetime.utcnow().isoformat(),
                "result": None,
                "error": None
            }
    
    def update_progress(self, training_id: str, progress: int, message: str):
        with self._lock:
            if training_id in self._status:
                self._status[training_id]["progress"] = progress
                self._status[training_id]["message"] = message
    
    def complete_training(self, training_id: str, result: Dict):
        with self._lock:
            if training_id in self._status:
                self._status[training_id]["status"] = "completed"
                self._status[training_id]["progress"] = 100
                self._status[training_id]["result"] = result
    
    def fail_training(self, training_id: str, error: str):
        with self._lock:
            if training_id in self._status:
                self._status[training_id]["status"] = "failed"
                self._status[training_id]["error"] = error
    
    def get_status(self, training_id: str) -> Optional[Dict]:
        with self._lock:
            return self._status.get(training_id)
    
    def remove_status(self, training_id: str):
        with self._lock:
            if training_id in self._status:
                del self._status[training_id]

training_status = TrainingStatus()

class Trainer:
    def __init__(self):
        self._training_lock = threading.Lock()
        self._is_training = False
    
    def is_training(self) -> bool:
        return self._is_training
    
    def _train_async(self, training_id: str, tfidf_config: Dict = None, base_snapshot_id: int = None):
        try:
            self._is_training = True
            logger.info(f"Starting training {training_id}")
            
            training_status.update_progress(training_id, 5, "Loading training data...")
            texts, labels, category_ids = data_loader.get_labeled_data()
            
            if len(texts) == 0:
                raise ValueError("No training data available. Please import data first.")
            
            categories = data_loader.get_categories()
            category_names = [cat.name for cat in categories]
            category_id_map = {cat.id: cat.name for cat in categories}
            
            training_status.update_progress(training_id, 15, "Splitting train/test sets...")
            X_train_texts, X_test_texts, y_train, y_test = train_test_split(
                texts, labels,
                test_size=Config.TEST_SIZE,
                random_state=Config.RANDOM_STATE,
                stratify=labels
            )
            
            training_status.update_progress(training_id, 30, "Extracting TF-IDF features...")
            if base_snapshot_id is not None:
                db = get_db_session()
                try:
                    base_snapshot = db.query(ModelSnapshot).filter(ModelSnapshot.id == base_snapshot_id).first()
                    if base_snapshot and os.path.exists(base_snapshot.vectorizer_path):
                        vectorizer = feature_engineer.load_vectorizer(base_snapshot.vectorizer_path)
                        X_train = feature_engineer.transform(X_train_texts, vectorizer)
                        X_test = feature_engineer.transform(X_test_texts, vectorizer)
                    else:
                        X_train, vectorizer = feature_engineer.fit_transform(X_train_texts, tfidf_config)
                        X_test = feature_engineer.transform(X_test_texts, vectorizer)
                finally:
                    db.close()
            else:
                X_train, vectorizer = feature_engineer.fit_transform(X_train_texts, tfidf_config)
                X_test = feature_engineer.transform(X_test_texts, vectorizer)
            
            training_status.update_progress(training_id, 50, "Training Naive Bayes classifier...")
            label_encoder = LabelEncoder()
            label_encoder.fit(category_names)
            
            y_train_encoded = label_encoder.transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            classes = label_encoder.classes_.tolist()
            
            model = MultinomialNB()
            
            if base_snapshot_id is not None:
                db = get_db_session()
                try:
                    base_snapshot = db.query(ModelSnapshot).filter(ModelSnapshot.id == base_snapshot_id).first()
                    if base_snapshot and os.path.exists(base_snapshot.model_path):
                        base_model = joblib.load(base_snapshot.model_path)
                        model = base_model
                        model.partial_fit(X_train, y_train_encoded, classes=np.unique(y_train_encoded))
                        logger.info(f"Incremental learning from snapshot {base_snapshot_id}")
                    else:
                        model.fit(X_train, y_train_encoded)
                finally:
                    db.close()
            else:
                model.fit(X_train, y_train_encoded)
            
            training_status.update_progress(training_id, 70, "Evaluating model...")
            y_pred_encoded = model.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            y_test_labels = label_encoder.inverse_transform(y_test_encoded)
            
            metrics = evaluator.evaluate(y_test_labels, y_pred, classes)
            
            training_status.update_progress(training_id, 85, "Saving model snapshot...")
            snapshot_name = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            model_path = os.path.join(Config.MODEL_DIR, f"{snapshot_name}.joblib")
            vectorizer_path = os.path.join(Config.MODEL_DIR, f"{snapshot_name}_vectorizer.joblib")
            
            joblib.dump(model, model_path)
            feature_engineer.save_vectorizer(vectorizer, vectorizer_path)
            
            default_cat_id = categories[0].id if categories else None
            
            db = get_db_session()
            try:
                db.query(ModelSnapshot).update({ModelSnapshot.is_active: 0})
                
                snapshot = ModelSnapshot(
                    name=snapshot_name,
                    category_id=default_cat_id,
                    model_path=model_path,
                    vectorizer_path=vectorizer_path,
                    accuracy=metrics["accuracy"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1_score=metrics["f1_score"],
                    confusion_matrix=metrics["confusion_matrix"],
                    tfidf_config=tfidf_config or Config.TFIDF_DEFAULT_CONFIG,
                    train_sample_count=len(X_train_texts),
                    test_sample_count=len(X_test_texts),
                    is_active=1
                )
                db.add(snapshot)
                db.commit()
                db.refresh(snapshot)
                snapshot_id = snapshot.id
            finally:
                db.close()
            
            training_status.update_progress(training_id, 95, "Loading model for prediction...")
            predictor.load_model(model, vectorizer, classes, snapshot_id)
            
            self._cleanup_old_snapshots()
            
            result = {
                "snapshot_id": snapshot_id,
                "snapshot_name": snapshot_name,
                "model_path": model_path,
                "vectorizer_path": vectorizer_path,
                "train_sample_count": len(X_train_texts),
                "test_sample_count": len(X_test_texts),
                "metrics": metrics,
                "classes": classes
            }
            
            training_status.complete_training(training_id, result)
            logger.info(f"Training {training_id} completed successfully, snapshot_id: {snapshot_id}")
            
        except Exception as e:
            logger.error(f"Training {training_id} failed: {e}", exc_info=True)
            training_status.fail_training(training_id, str(e))
        finally:
            self._is_training = False
    
    def train(self, tfidf_config: Dict = None, base_snapshot_id: int = None) -> str:
        with self._training_lock:
            if self._is_training:
                raise RuntimeError("Another training is already in progress")
            
            training_id = f"train_{int(time.time() * 1000)}"
            training_status.init_training(training_id)
            
            thread = threading.Thread(
                target=self._train_async,
                args=(training_id, tfidf_config, base_snapshot_id),
                daemon=True
            )
            thread.start()
            
            return training_id
    
    def _cleanup_old_snapshots(self):
        db = get_db_session()
        try:
            snapshots = db.query(ModelSnapshot).order_by(ModelSnapshot.trained_at.desc()).all()
            
            if len(snapshots) > Config.MAX_SNAPSHOTS:
                snapshots_to_delete = snapshots[Config.MAX_SNAPSHOTS:]
                
                for snapshot in snapshots_to_delete:
                    for path in [snapshot.model_path, snapshot.vectorizer_path]:
                        if os.path.exists(path):
                            os.remove(path)
                            logger.info(f"Deleted old snapshot file: {path}")
                    
                    if snapshot.is_active == 1 and len(snapshots) > 1:
                        next_active = snapshots[Config.MAX_SNAPSHOTS - 1]
                        next_active.is_active = 1
                    
                    db.delete(snapshot)
                
                db.commit()
                logger.info(f"Cleaned up {len(snapshots_to_delete)} old snapshots")
        except Exception as e:
            logger.error(f"Error cleaning up snapshots: {e}")
            db.rollback()
        finally:
            db.close()

trainer = Trainer()
