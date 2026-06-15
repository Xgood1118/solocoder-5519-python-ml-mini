from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import Config
from app.utils.logger import logger

engine = create_engine(f"sqlite:///{Config.DB_PATH}", echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from app.models import Category, Document, ModelSnapshot
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")

def get_db_session():
    return next(get_db())
