from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from config import Config
from app.utils.logger import logger

_base_engines = {}
_base_sessions = {}
Base = declarative_base()


def _get_engine():
    db_path = Config.DB_PATH
    if db_path not in _base_engines:
        connect_args = {"check_same_thread": False} if True else {}
        engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args=connect_args,
            poolclass=StaticPool
        )
        _base_engines[db_path] = engine
        _base_sessions[db_path] = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.debug(f"Created new DB engine for: {db_path}")
    return _base_engines[db_path]


def _get_session_local():
    _get_engine()
    return _base_sessions[Config.DB_PATH]


def __getattr__(name):
    if name == "engine":
        return _get_engine()
    elif name == "SessionLocal":
        return _get_session_local()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_db():
    db = _get_session_local()()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app.models import Category, Document, ModelSnapshot
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database initialized: {Config.DB_PATH}")


def get_db_session():
    return next(get_db())
