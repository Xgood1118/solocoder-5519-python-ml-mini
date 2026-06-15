import os
import json
import pandas as pd
from typing import List, Tuple
from sqlalchemy.orm import Session
from app.models import Category, Document, get_db_session
from app.services.preprocessor import preprocessor
from app.utils.logger import logger

class DataLoader:
    @staticmethod
    def load_csv(file_path: str) -> List[Tuple[str, str]]:
        logger.info(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path, header=None, names=["text", "category"], dtype=str)
        df = df.dropna(subset=["text", "category"])
        data = list(zip(df["text"].tolist(), df["category"].tolist()))
        logger.info(f"Loaded {len(data)} samples from CSV")
        return data
    
    @staticmethod
    def load_json(file_path: str) -> List[Tuple[str, str]]:
        logger.info(f"Loading JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        
        if isinstance(data_list, dict) and "data" in data_list:
            data_list = data_list["data"]
        
        data = []
        for item in data_list:
            text = item.get("text") or item.get("content")
            category = item.get("category") or item.get("label")
            if text and category:
                data.append((str(text), str(category)))
        
        logger.info(f"Loaded {len(data)} samples from JSON")
        return data
    
    @staticmethod
    def load_file(file_path: str) -> List[Tuple[str, str]]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return DataLoader.load_csv(file_path)
        elif ext == ".json":
            return DataLoader.load_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    @staticmethod
    def save_to_db(data: List[Tuple[str, str]], source_file: str = None) -> dict:
        db = get_db_session()
        try:
            category_stats = {}
            total_imported = 0
            
            for text, category_name in data:
                category = db.query(Category).filter(Category.name == category_name).first()
                if not category:
                    category = Category(
                        name=category_name,
                        display_name=category_name,
                        sample_count=0
                    )
                    db.add(category)
                    db.flush()
                
                processed_text = preprocessor.tokenize_for_tfidf(text)
                
                doc = Document(
                    text=text,
                    category_id=category.id,
                    source_file=source_file
                )
                db.add(doc)
                
                category.sample_count += 1
                category_stats[category_name] = category_stats.get(category_name, 0) + 1
                total_imported += 1
            
            db.commit()
            logger.info(f"Imported {total_imported} documents into {len(category_stats)} categories")
            return {
                "total_imported": total_imported,
                "category_distribution": category_stats
            }
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving to database: {e}")
            raise
        finally:
            db.close()
    
    @staticmethod
    def get_labeled_data() -> Tuple[List[str], List[str], List[int]]:
        db = get_db_session()
        try:
            documents = db.query(Document).all()
            texts = []
            labels = []
            category_ids = []
            
            for doc in documents:
                texts.append(doc.text)
                labels.append(doc.category.name)
                category_ids.append(doc.category_id)
            
            logger.info(f"Retrieved {len(texts)} labeled documents")
            return texts, labels, category_ids
        finally:
            db.close()
    
    @staticmethod
    def get_categories() -> List[Category]:
        db = get_db_session()
        try:
            return db.query(Category).all()
        finally:
            db.close()

data_loader = DataLoader()
