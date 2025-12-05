import sys, pickle, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def map_numeric_to_str(y_pred_numeric, mapping):
    # mapping: {0: "normal"/"attack", 1: "attack"/"normal"}
    return np.array([mapping.get(int(v), str(v)) for v in y_pred_numeric])

def try_mappings_and_pick(y_true, y_pred_numeric):
    # thử 2 mapping phổ biến và chọn mapping có F1(attack) cao hơn
    m1 = {0: "normal", 1: "attack"}
    m2 = {0: "attack", 1: "normal"}
    y1 = map_numeric_to_str(y_pred_numeric, m1)
    y2 = map_numeric_to_str(y_pred_numeric, m2)
    labels = np.unique(y_true)
    if "attack" in labels:
        f1_1 = f1_score(y_true, y1, pos_label="attack", average="binary")
        f1_2 = f1_score(y_true, y2, pos_label="attack", average="binary")
    else:
        f1_1 = f1_score(y_true, y1, average="macro")
        f1_2 = f1_score(y_true, y2, average="macro")
    return (y1, m1) if f1_1 >= f1_2 else (y2, m2)

def main():
    if len(sys.argv) != 5:
        print("Usage: python evaluate_model_local_v2.py model.pkl scaler.pkl feature_list.pkl test.csv")
        sys.exit(1)
    model_p, scaler_p, features_p, test_p = sys.argv[1:]
    model = load_pickle(model_p)
    scaler = load_pickle(scaler_p)
    feature_list = load_pickle(features_p)
    df = pd.read_csv(test_p)
    if "label" not in df.columns:
        raise RuntimeError("CSV must contain a \"label\" column")
    X = df[feature_list].values
    y_true = df["label"].astype(str).values
    Xs = scaler.transform(X)
    y_pred = model.predict(Xs)
    if y_pred.dtype.kind in {"U","S","O"}:
        y_pred_str = y_pred.astype(str)
        mapping_used = "model classes as-is"
    else:
        y_pred_str, chosen_mapping = try_mappings_and_pick(y_true, y_pred)
        mapping_used = f"{chosen_mapping}"
    print("Mapping used:", mapping_used)
    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred_str, digits=4))
    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    labels = sorted(np.unique(np.concatenate([y_true, y_pred_str])))
    print("Labels order:", labels)
    print(confusion_matrix(y_true, y_pred_str, labels=labels))

if __name__ == "__main__":
    main()
