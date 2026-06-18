import numpy as np

try:
    from sklearn.metrics import f1_score
    def compute_macro_f1(preds, targets):
        return f1_score(targets, preds, average='macro', zero_division=0)
except ImportError:
    def compute_macro_f1(preds, targets):
        classes = np.unique(np.concatenate([preds, targets]))
        if len(classes) == 0:
            return 0.0
        f1s = []
        for c in classes:
            tp = np.sum((preds == c) & (targets == c))
            fp = np.sum((preds == c) & (targets != c))
            fn = np.sum((preds != c) & (targets == c))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1s.append(f1)
        return np.mean(f1s) if len(f1s) > 0 else 0.0
