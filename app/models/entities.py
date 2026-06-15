from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Text, JSON
from sqlalchemy.orm import relationship
from app.models.database import Base

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200))
    sample_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    documents = relationship("Document", back_populates="category", cascade="all, delete-orphan")
    snapshots = relationship("ModelSnapshot", back_populates="category", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "sample_count": self.sample_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id", ondelete="CASCADE"), nullable=False, index=True)
    source_file = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    category = relationship("Category", back_populates="documents")
    
    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "category_id": self.category_id,
            "category_name": self.category.name if self.category else None,
            "source_file": self.source_file,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class ModelSnapshot(Base):
    __tablename__ = "model_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200))
    category_id = Column(Integer, ForeignKey("categories.id", ondelete="CASCADE"), nullable=False, index=True)
    model_path = Column(String(500), nullable=False)
    vectorizer_path = Column(String(500), nullable=False)
    trained_at = Column(DateTime, default=datetime.utcnow, index=True)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    confusion_matrix = Column(JSON)
    tfidf_config = Column(JSON)
    train_sample_count = Column(Integer)
    test_sample_count = Column(Integer)
    is_active = Column(Integer, default=0)
    
    category = relationship("Category", back_populates="snapshots")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "category_id": self.category_id,
            "model_path": self.model_path,
            "vectorizer_path": self.vectorizer_path,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix,
            "tfidf_config": self.tfidf_config,
            "train_sample_count": self.train_sample_count,
            "test_sample_count": self.test_sample_count,
            "is_active": self.is_active == 1
        }
