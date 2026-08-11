import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def evaluate_classifier(y_true, y_pred, y_probs=None, class_names=None):
    """
    Computes complete classification metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC (if probabilities provided)
    - Confusion Matrix
    - Per-class classification report
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    auc = None
    if y_probs is not None:
        try:
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_probs)
            else:
                auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except Exception:
            auc = None
            
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else "N/A",
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

if __name__ == "__main__":
    y_true = [0, 1, 1, 0, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 0, 0, 0, 1, 1]
    y_probs = [0.1, 0.9, 0.85, 0.2, 0.45, 0.15, 0.95, 0.88]
    res = evaluate_classifier(y_true, y_pred, y_probs, class_names=['Real', 'Fake'])
    print("Sample Evaluation Metric Output:", res)
