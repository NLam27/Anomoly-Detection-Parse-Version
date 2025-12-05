# post_samples_to_api.py
# Usage: python post_samples_to_api.py test_samples.csv http://127.0.0.1:8000/api/predict/
import sys, time, pandas as pd, requests

def main():
    if len(sys.argv) < 2:
        print("Usage: python post_samples_to_api.py test_samples.csv [url]")
        return
    csv_path = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) >= 3 else "http://127.0.0.1:8000/api/predict/"
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        payload = row.to_dict()
        payload.pop('label', None)
        try:
            r = requests.post(url, json=payload, timeout=5)
            print(i, "status", r.status_code, r.json())
        except Exception as e:
            print(i, "error", e)
        time.sleep(0.05)

if __name__ == "__main__":
    main()
