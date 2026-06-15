import os
from flask import Flask, request, jsonify, Blueprint
from werkzeug.utils import secure_filename

from config import Config
from app.models import init_db, Category, Document, get_db_session
from app.services import (
    data_loader, trainer, training_status, predictor,
    evaluator, snapshot_manager
)
from app.utils import logger, training_limiter

api_bp = Blueprint("api", __name__)

def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    
    init_db()
    
    logger.info("Loading latest model snapshot...")
    snapshot_manager.load_latest_snapshot()
    
    app.register_blueprint(api_bp, url_prefix="/api")
    
    @app.route("/")
    def index():
        return jsonify({
            "name": "ML Toy Project - Text Classification API",
            "version": "1.0.0",
            "status": "running",
            "model_ready": predictor.is_ready(),
            "endpoints": {
                "data_import": "POST /api/data/import",
                "data_list": "GET /api/data/documents",
                "categories_list": "GET /api/categories",
                "train_start": "POST /api/train",
                "train_status": "GET /api/train/status/<training_id>",
                "train_incremental": "POST /api/train/incremental",
                "predict_single": "POST /api/predict",
                "predict_batch": "POST /api/predict/batch",
                "snapshot_list": "GET /api/snapshots",
                "snapshot_load": "POST /api/snapshots/<snapshot_id>/load",
                "snapshot_compare": "GET /api/snapshots/compare",
                "snapshot_delete": "DELETE /api/snapshots/<snapshot_id>"
            }
        })
    
    return app

@api_bp.route("/data/import", methods=["POST"])
def import_data():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        filename = secure_filename(file.filename)
        temp_path = os.path.join(Config.DATA_DIR, filename)
        file.save(temp_path)
        
        try:
            data = data_loader.load_file(temp_path)
            result = data_loader.save_to_db(data, source_file=filename)
            
            return jsonify({
                "success": True,
                "message": f"Successfully imported {result['total_imported']} samples",
                "data": result
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"Data import failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/data/documents", methods=["GET"])
def list_documents():
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 20))
        
        db = get_db_session()
        try:
            query = db.query(Document)
            total = query.count()
            documents = query.order_by(Document.created_at.desc())\
                .offset((page - 1) * page_size)\
                .limit(page_size)\
                .all()
            
            return jsonify({
                "success": True,
                "data": {
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "documents": [doc.to_dict() for doc in documents]
                }
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/categories", methods=["GET"])
def list_categories():
    try:
        db = get_db_session()
        try:
            categories = db.query(Category).all()
            return jsonify({
                "success": True,
                "data": [cat.to_dict() for cat in categories]
            })
        finally:
            db.close()
    except Exception as e:
        logger.error(f"List categories failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/train", methods=["POST"])
@training_limiter.limit("training")
def start_training():
    try:
        if trainer.is_training():
            return jsonify({
                "success": False,
                "error": "Training is already in progress"
            }), 429
        
        tfidf_config = request.json.get("tfidf_config") if request.is_json else None
        
        training_id = trainer.train(tfidf_config=tfidf_config)
        
        return jsonify({
            "success": True,
            "message": "Training started",
            "data": {
                "training_id": training_id,
                "status_url": f"/api/train/status/{training_id}"
            }
        })
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 429
    except Exception as e:
        logger.error(f"Start training failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/train/incremental", methods=["POST"])
@training_limiter.limit("training")
def start_incremental_training():
    try:
        if trainer.is_training():
            return jsonify({
                "success": False,
                "error": "Training is already in progress"
            }), 429
        
        data = request.json or {}
        base_snapshot_id = data.get("base_snapshot_id")
        tfidf_config = data.get("tfidf_config")
        
        if base_snapshot_id is None:
            latest = snapshot_manager.get_latest_snapshot()
            if latest:
                base_snapshot_id = latest.id
        
        training_id = trainer.train(
            tfidf_config=tfidf_config,
            base_snapshot_id=base_snapshot_id
        )
        
        return jsonify({
            "success": True,
            "message": "Incremental training started",
            "data": {
                "training_id": training_id,
                "base_snapshot_id": base_snapshot_id,
                "status_url": f"/api/train/status/{training_id}"
            }
        })
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e)}), 429
    except Exception as e:
        logger.error(f"Incremental training failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/train/status/<training_id>", methods=["GET"])
def get_training_status(training_id):
    try:
        status = training_status.get_status(training_id)
        if not status:
            return jsonify({"success": False, "error": "Training not found"}), 404
        
        return jsonify({
            "success": True,
            "data": status
        })
    except Exception as e:
        logger.error(f"Get training status failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/predict", methods=["POST"])
def predict_single():
    try:
        if not predictor.is_ready():
            return jsonify({
                "success": False,
                "error": "Model not ready. Please train a model first."
            }), 503
        
        data = request.json or {}
        text = data.get("text")
        top_n = data.get("top_n")
        
        if not text:
            return jsonify({"success": False, "error": "Text is required"}), 400
        
        result = predictor.predict_single(text, top_n=top_n)
        
        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        if not predictor.is_ready():
            return jsonify({
                "success": False,
                "error": "Model not ready. Please train a model first."
            }), 503
        
        data = request.json or {}
        texts = data.get("texts", [])
        top_n = data.get("top_n")
        
        if not texts or not isinstance(texts, list):
            return jsonify({"success": False, "error": "Texts list is required"}), 400
        
        results = predictor.predict_batch(texts, top_n=top_n)
        
        return jsonify({
            "success": True,
            "data": {
                "count": len(results),
                "predictions": results
            }
        })
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/snapshots", methods=["GET"])
def list_snapshots():
    try:
        snapshots = snapshot_manager.get_all_snapshots()
        return jsonify({
            "success": True,
            "data": [snap.to_dict() for snap in snapshots]
        })
    except Exception as e:
        logger.error(f"List snapshots failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/snapshots/<int:snapshot_id>/load", methods=["POST"])
def load_snapshot(snapshot_id):
    try:
        snapshot_manager.load_snapshot(snapshot_id)
        return jsonify({
            "success": True,
            "message": f"Snapshot {snapshot_id} loaded successfully"
        })
    except Exception as e:
        logger.error(f"Load snapshot failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/snapshots/compare", methods=["GET"])
def compare_snapshots():
    try:
        snapshots = snapshot_manager.get_all_snapshots()
        comparison = evaluator.compare_snapshots(snapshots)
        return jsonify({
            "success": True,
            "data": comparison
        })
    except Exception as e:
        logger.error(f"Compare snapshots failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/snapshots/<int:snapshot_id>", methods=["DELETE"])
def delete_snapshot(snapshot_id):
    try:
        success = snapshot_manager.delete_snapshot(snapshot_id)
        if not success:
            return jsonify({"success": False, "error": "Snapshot not found"}), 404
        
        return jsonify({
            "success": True,
            "message": f"Snapshot {snapshot_id} deleted successfully"
        })
    except Exception as e:
        logger.error(f"Delete snapshot failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
