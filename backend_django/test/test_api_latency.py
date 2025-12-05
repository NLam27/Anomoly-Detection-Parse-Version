# test_api_latency.py

import pandas as pd
import requests
import time
from tqdm import tqdm

# Cấu hình
API_URL = "http://localhost:8000/api/predict/"
FEATURES_FILE = "flows_to_test.csv"  # file CSV chứa các dòng đặc trưng
BATCH_SIZE = 100  # số lượng request gửi mỗi lần (có thể bỏ hoặc dùng thì thêm logic)
SLEEP_BETWEEN_BATCHES = 1.0  # giây nghỉ giữa các batch (nếu dùng batch logic)

def load_features(file_path):
    df = pd.read_csv(file_path)
    # Nếu file csv có cả nhãn, loại bỏ nhãn
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    return df

def send_one(payload):
    resp = requests.post(API_URL, json=payload, timeout=5)
    return resp.status_code, resp.json()

def run_test(df):
    total = len(df)
    latencies = []
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=total):
        payload = row.to_dict()
        start = time.time()
        try:
            status, result = send_one(payload)
        except Exception as ex:
            status = None
            result = None
            errors += 1
        latency = time.time() - start
        latencies.append(latency)

    # Tóm tắt kết quả
    latencies = [l for l in latencies if l is not None]
    print(f"Sent {total} requests, errors: {errors}")
    print(f"Average latency: {sum(latencies)/len(latencies):.3f} s")
    print(f"Max latency: {max(latencies):.3f} s")
    print(f"Min latency: {min(latencies):.3f} s")
    throughput = len(latencies) / sum(latencies)
    print(f"Estimated throughput: {throughput:.1f} requests/sec")

if __name__ == "__main__":
    df = load_features(FEATURES_FILE)
    run_test(df)
