import os
import joblib
from typing import Optional, List
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from config import Config
from app.models import ModelSnapshot, Category, get_db_session
from app.services.predictor import predictor
from app.utils.logger import logger

class SnapshotManager:
    @staticmethod
    def get_latest_snapshot() -> Optional[ModelSnapshot]:
        db = get_db_session()
        try:
            snapshot = db.query(ModelSnapshot).filter(ModelSnapshot.is_active == 1).first()
            if not snapshot:
                snapshot = db.query(ModelSnapshot).order_by(ModelSnapshot.trained_at.desc()).first()
            return snapshot
        finally:
            db.close()
    
    @staticmethod
    def get_snapshot(snapshot_id: int) -> Optional[ModelSnapshot]:
        db = get_db_session()
        try:
            return db.query(ModelSnapshot).filter(ModelSnapshot.id == snapshot_id).first()
        finally:
            db.close()
    
    @staticmethod
    def get_all_snapshots() -> List[ModelSnapshot]:
        db = get_db_session()
        try:
            return db.query(ModelSnapshot).order_by(ModelSnapshot.trained_at.desc()).all()
        finally:
            db.close()
    
    @staticmethod
    def load_snapshot(snapshot_id: int) -> bool:
        db = get_db_session()
        try:
            snapshot = db.query(ModelSnapshot).filter(ModelSnapshot.id == snapshot_id).first()
            if not snapshot:
                raise ValueError(f"Snapshot {snapshot_id} not found")
            
            if not os.path.exists(snapshot.model_path):
                raise FileNotFoundError(f"Model file not found: {snapshot.model_path}")
            if not os.path.exists(snapshot.vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found: {snapshot.vectorizer_path}")
            
            model: MultinomialNB = joblib.load(snapshot.model_path)
            vectorizer: TfidfVectorizer = joblib.load(snapshot.vectorizer_path)
            
            categories = db.query(Category).all()
            category_names = [cat.name for cat in categories]
            
            label_encoder = LabelEncoder()
            label_encoder.fit(category_names)
            classes = label_encoder.classes_.tolist()
            
            if hasattr(model, "classes_"):
                model_classes = [category_names[i] for i in model.classes_]
                if set(model_classes) == set(classes):
                    classes = model_classes
            
            db.query(ModelSnapshot).update({ModelSnapshot.is_active: 0})
            snapshot.is_active = 1
            db.commit()
            
            predictor.load_model(model, vectorizer, classes, snapshot_id)
            logger.info(f"Loaded snapshot {snapshot_id} as active model")
            return True
        except Exception as e:
            logger.error(f"Error loading snapshot {snapshot_id}: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    @staticmethod
    def load_latest_snapshot() -> bool:
        snapshot = SnapshotManager.get_latest_snapshot()
        if snapshot:
            try:
                return SnapshotManager.load_snapshot(snapshot.id)
            except Exception as e:
                logger.warning(f"Failed to load latest snapshot: {e}")
        logger.info("No snapshot available to load")
        return False
    
    @staticmethod
    def delete_snapshot(snapshot_id: int) -> bool:
        db = get_db_session()
        try:
            snapshot = db.query(ModelSnapshot).filter(ModelSnapshot.id == snapshot_id).first()
            if not snapshot:
                return False
            
            for path in [snapshot.model_path, snapshot.vectorizer_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            db.delete(snapshot)
            db.commit()
            
            if snapshot.is_active == 1:
                latest = SnapshotManager.get_latest_snapshot()
                if latest:
                    latest.is_active = 1
                    db.commit()
            
            logger.info(f"Deleted snapshot {snapshot_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting snapshot {snapshot_id}: {e}")
            db.rollback()
            return False
        finally:
            db.close()

snapshot_manager = SnapshotManager()
