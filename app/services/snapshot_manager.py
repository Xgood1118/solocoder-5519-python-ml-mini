import os
import joblib
from typing import Optional, List
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

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
            
            logger.info(f"Loading snapshot {snapshot_id}: model={snapshot.model_path}, vec={snapshot.vectorizer_path}")
            
            model: MultinomialNB = joblib.load(snapshot.model_path)
            logger.info(f"  Model loaded, type={type(model).__name__}, classes_encoded={model.classes_.tolist()}")
            
            vectorizer: TfidfVectorizer = joblib.load(snapshot.vectorizer_path)
            logger.info(f"  Vectorizer loaded, vocab_size={len(vectorizer.vocabulary_)}")
            
            classes = snapshot.classes
            if not classes:
                logger.warning(f"  Snapshot has no classes field in DB, reconstructing from categories")
                categories = db.query(Category).all()
                category_names = [cat.name for cat in categories]
                classes = [category_names[i] for i in model.classes_]
                snapshot.classes = classes
                db.commit()
                logger.warning(f"  Reconstructed classes and saved back: {classes}")
            
            model_classes_names = [classes[i] for i in model.classes_]
            assert model_classes_names == classes, (
                f"Class order mismatch! model.classes_ -> names={model_classes_names} "
                f"vs snapshot.classes={classes}"
            )
            logger.info(f"  Classes verified OK: {classes}")
            
            db.query(ModelSnapshot).filter(ModelSnapshot.id != snapshot_id).update({ModelSnapshot.is_active: 0})
            snapshot.is_active = 1
            db.commit()
            
            predictor.load_model(model, vectorizer, classes, snapshot_id)
            logger.info(f"  Predictor ready={predictor.is_ready()}, snapshot_id={predictor.snapshot_id}")
            return True
        except AssertionError as e:
            logger.error(f"Load snapshot {snapshot_id} class mismatch: {e}")
            db.rollback()
            raise
        except Exception as e:
            logger.error(f"Error loading snapshot {snapshot_id}: {e}", exc_info=True)
            db.rollback()
            raise
        finally:
            db.close()
    
    @staticmethod
    def load_latest_snapshot() -> bool:
        snapshot = SnapshotManager.get_latest_snapshot()
        if snapshot:
            logger.info(f"Found latest snapshot, id={snapshot.id}, is_active={snapshot.is_active == 1}, classes={snapshot.classes}")
            try:
                return SnapshotManager.load_snapshot(snapshot.id)
            except Exception as e:
                logger.warning(f"Failed to load latest snapshot id={snapshot.id}: {e}")
        logger.info("No snapshot available to load (predictor will return 503 until training)")
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
