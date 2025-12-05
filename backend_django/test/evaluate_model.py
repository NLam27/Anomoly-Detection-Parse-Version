# evaluate_model.py
# Dùng để test locally: nạp model, nạp X_test/y_test (npz/csv), in classification_report + confusion matrix.
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support

def load_data(path):
    # Hỗ trợ .npz/.npy hoặc .csv
    if path.endswith('.npz') or path.endswith('.npz.npy') or path.endswith('.npy'):
        arr = np.load(path, allow_pickle=True)
        # trường hợp .npz chứa arrays 'X' 'y'
        if isinstance(arr, np.lib.npyio.NpzFile):
            if 'X' in arr and 'y' in arr:
                return arr['X'], arr['y']
            # fallback: load as dict-like
            keys = list(arr.files)
            if len(keys) >= 2:
                return arr[keys[0]], arr[keys[1]]
            raise RuntimeError("Không tìm được X/y trong .npz")
        else:
            # chỉ 1 mảng
            raise RuntimeError(".npy không đủ thông tin. Dùng .npz hoặc .csv")
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
        # giả sử label cột 'label'
        if 'label' not in df.columns:
            raise RuntimeError("CSV cần có cột 'label'")
        y = df['label'].values
        X = df.drop(columns=['label']).values
        return X, y
    else:
        raise RuntimeError("Định dạng file không hỗ trợ. Dùng .npz hoặc .csv")

def main():
    if len(sys.argv) < 3:
        print("Usage: python evaluate_model.py model_path test_data_path")
        sys.exit(1)
    model_path = sys.argv[1]
    test_path = sys.argv[2]
    print("Load model:", model_path)
    model = joblib.load(model_path)
    print("Load test data:", test_path)
    X_test, y_test = load_data(test_path)
    print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

    y_pred = model.predict(X_test)
    # handle predict_proba for multiclass if exists
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            proba = None

    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))
    # Per-class precision/recall/f1
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    print("\nPer-class P/R/F:")
    for i, (pp, rr, ff) in enumerate(zip(p, r, f)):
        print(f"Class {i}: P={pp:.4f}, R={rr:.4f}, F1={ff:.4f}")

    # (Optional) ROC AUC if binary
    if proba is not None and len(np.unique(y_test)) == 2:
        try:
            auc = roc_auc_score(y_test, proba[:,1])
            print("\nROC AUC:", auc)
        except Exception as e:
            print("Không tính được ROC AUC:", e)

if __name__ == '__main__':
    main()
