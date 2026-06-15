from waitress import serve
from app.api import create_app
from config import Config
from app.utils.logger import logger

app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting ML Toy Project server on {Config.HOST}:{Config.PORT}")
    logger.info("Server mode: production (waitress)")
    serve(
        app,
        host=Config.HOST,
        port=Config.PORT,
        threads=8
    )
