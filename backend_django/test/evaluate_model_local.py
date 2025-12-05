# evaluate_model_local.py
# Usage:
#   python evaluate_model_local.py model.pkl scaler.pkl feature_list.pkl test.csv
import sys, pickle, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    if len(sys.argv) != 5:
        print("Usage: python evaluate_model_local.py model.pkl scaler.pkl feature_list.pkl test.csv")
        sys.exit(1)
    model_p, scaler_p, features_p, test_p = sys.argv[1:]
    model = load_pickle(model_p)
    scaler = load_pickle(scaler_p)
    feature_list = load_pickle(features_p)

    if test_p.endswith(".csv"):
        df = pd.read_csv(test_p)
        if 'label' not in df.columns:
            raise RuntimeError("CSV must contain a 'label' column")
        X = df[feature_list].values
        y = df['label'].values
    else:
        data = np.load(test_p, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            X, y = data['X'], data['y']
        else:
            raise RuntimeError("Unsupported file; use CSV or .npz with X/y")

    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)

    print("\n=== Classification report ===")
    print(classification_report(y, y_pred, digits=4))
    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y, y_pred))

    p, r, f, _ = precision_recall_fscore_support(y, y_pred, average=None)
    labels = sorted(np.unique(y))
    print("\nPer-class (precision / recall / f1):")
    for lab, pp, rr, ff in zip(labels, p, r, f):
        print(f"Label {lab}: P={pp:.4f} R={rr:.4f} F1={ff:.4f}")

if __name__ == "__main__":
    main()
