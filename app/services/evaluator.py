from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from typing import Dict, List
from app.utils.logger import logger

class ModelEvaluator:
    @staticmethod
    def evaluate(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
        logger.info(f"Evaluating model on {len(y_true)} samples with {len(labels)} classes")
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_dict = ModelEvaluator._confusion_matrix_to_dict(cm, labels)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm_dict
        }
        
        logger.info(f"Evaluation results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return metrics
    
    @staticmethod
    def _confusion_matrix_to_dict(cm: np.ndarray, labels: List[str]) -> Dict:
        cm_list = cm.tolist()
        
        per_class_metrics = {}
        for i, label in enumerate(labels):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(len(labels)) if j != i)
            fn = sum(cm[i][j] for j in range(len(labels)) if j != i)
            tn = sum(cm[j][k] for j in range(len(labels)) for k in range(len(labels)) if j != i and k != i)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[label] = {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
        
        return {
            "matrix": cm_list,
            "labels": labels,
            "per_class": per_class_metrics
        }
    
    @staticmethod
    def compare_snapshots(snapshots: List) -> List[Dict]:
        comparison = []
        for snap in snapshots:
            comparison.append({
                "snapshot_id": snap.id,
                "name": snap.name,
                "trained_at": snap.trained_at.isoformat() if snap.trained_at else None,
                "accuracy": snap.accuracy,
                "precision": snap.precision,
                "recall": snap.recall,
                "f1_score": snap.f1_score,
                "tfidf_config": snap.tfidf_config,
                "train_sample_count": snap.train_sample_count,
                "test_sample_count": snap.test_sample_count,
                "is_active": snap.is_active == 1
            })
        return comparison

evaluator = ModelEvaluator()
