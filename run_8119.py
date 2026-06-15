import sys
sys.path.insert(0, '.')

from config import Config
Config.PORT = 8119
Config.HOST = '127.0.0.1'

from waitress import serve
from app.api import create_app
from app.utils.logger import logger

app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting ML Toy Project server on {Config.HOST}:{Config.PORT}")
    serve(app, host=Config.HOST, port=Config.PORT, threads=8)
