import sys
import time
import logging
import requests
import numpy as np
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
from threading import Thread, Lock

# ================= CẤU HÌNH =================
API_URL = "http://127.0.0.1:8000/api/analyze/"  # Địa chỉ Django API của bạn
INTERFACE = "wlan0"  # Tên card mạng (Windows: "Wi-Fi", Linux: "eth0" hoặc "wlan0")
CAPTURE_WINDOW = 3.0  # Chu kỳ gom gói tin để phân tích (giây)
# ============================================

# Cấu hình log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlowStats:
    def __init__(self):
        self.start_time = time.time()
        self.packets = []  # List of (direction, size, timestamp, flags, header_len)
        # Direction: 1 = Forward (Client -> Server), -1 = Backward

    def add_packet(self, size, direction, flags, header_len):
        self.packets.append({
            'size': size,
            'time': time.time(),
            'direction': direction,
            'flags': flags,
            'header_len': header_len
        })

    def get_features(self):
        """Tính toán 24 features mà Model yêu cầu từ danh sách gói tin thô"""
        if not self.packets:
            return None

        # Tách dữ liệu
        sizes = np.array([p['size'] for p in self.packets])
        timestamps = np.array([p['time'] for p in self.packets])
        directions = np.array([p['direction'] for p in self.packets])
        
        fwd_mask = (directions == 1)
        bwd_mask = (directions == -1)
        
        fwd_sizes = sizes[fwd_mask]
        bwd_sizes = sizes[bwd_mask]
        
        # Tính toán thời gian (Duration & IAT)
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.001
        if duration == 0: duration = 0.001
        
        # Calculate IAT (Inter-Arrival Time)
        iats = np.diff(timestamps) if len(timestamps) > 1 else [0]
        fwd_iats = np.diff(timestamps[fwd_mask]) if len(timestamps[fwd_mask]) > 1 else [0]

        # === MAP FEATURES (Phải khớp chính xác tên key trong views.py) ===
        features = {
            # Backward Packet Stats
            'Bwd Packet Length Min': float(np.min(bwd_sizes)) if len(bwd_sizes) > 0 else 0,
            'Bwd Packet Length Std': float(np.std(bwd_sizes)) if len(bwd_sizes) > 0 else 0,
            'Bwd Packet Length Mean': float(np.mean(bwd_sizes)) if len(bwd_sizes) > 0 else 0,
            'Bwd Packet Length Max': float(np.max(bwd_sizes)) if len(bwd_sizes) > 0 else 0,
            'Bwd Packets/s': len(bwd_sizes) / duration,
            'Avg Bwd Segment Size': float(np.mean(bwd_sizes)) if len(bwd_sizes) > 0 else 0,
            'Bwd Header Length': sum([p['header_len'] for p in self.packets if p['direction'] == -1]),

            # Forward Packet Stats
            'Fwd Packet Length Max': float(np.max(fwd_sizes)) if len(fwd_sizes) > 0 else 0,
            'Fwd Packet Length Mean': float(np.mean(fwd_sizes)) if len(fwd_sizes) > 0 else 0,
            'Fwd Header Length': sum([p['header_len'] for p in self.packets if p['direction'] == 1]),
            'Fwd Header Length 1': sum([p['header_len'] for p in self.packets if p['direction'] == 1]), # Duplicate key fix
            'Total Length of Fwd Packets': float(np.sum(fwd_sizes)) if len(fwd_sizes) > 0 else 0,
            'min seg size forward': 32, # Default TCP min header

            # Flow Stats
            'Flow Bytes/s': np.sum(sizes) / duration,
            'Flow IAT Mean': float(np.mean(iats)) if len(iats) > 0 else 0,
            'Fwd IAT Min': float(np.min(fwd_iats)) if len(fwd_iats) > 0 else 0,
            
            # General Packet Stats
            'Packet Length Mean': float(np.mean(sizes)),
            'Packet Length Std': float(np.std(sizes)),
            'Packet Length Variance': float(np.var(sizes)),
            'Average Packet Size': float(np.mean(sizes)),
            
            # Flags & Windows (Simplified extraction)
            'Fwd PSH Flags': sum([1 for p in self.packets if p['direction'] == 1 and 'P' in p['flags']]),
            'PSH Flag Count': sum([1 for p in self.packets if 'P' in p['flags']]),
            'Init Win bytes forward': 0, # Cần deep inspection, tạm thời để 0 hoặc random nhỏ
            'Init Win bytes backward': 0,
        }
        return features

class TrafficMonitor:
    def __init__(self):
        self.active_flows = {} # Key: (src_ip, dst_ip, src_port, dst_port, proto)
        self.lock = Lock()
        self.running = True

    def packet_callback(self, packet):
        if not packet.haslayer(IP):
            return

        try:
            # Lấy thông tin cơ bản
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            proto = packet[IP].proto
            length = len(packet)
            
            src_port = 0
            dst_port = 0
            flags = ""
            header_len = 0

            if packet.haslayer(TCP):
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                flags = str(packet[TCP].flags)
                header_len = packet[TCP].dataofs * 4
            elif packet.haslayer(UDP):
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
                header_len = 8

            # Xác định hướng (Flow Direction)
            # Quy ước: IP nhỏ hơn là Client (chỉ để gom nhóm)
            if src_ip < dst_ip:
                flow_key = (src_ip, dst_ip, src_port, dst_port, proto)
                direction = 1 # Fwd
            else:
                flow_key = (dst_ip, src_ip, dst_port, src_port, proto)
                direction = -1 # Bwd

            with self.lock:
                if flow_key not in self.active_flows:
                    self.active_flows[flow_key] = FlowStats()
                self.active_flows[flow_key].add_packet(length, direction, flags, header_len)

        except Exception as e:
            pass

    def analyzer_loop(self):
        """Định kỳ quét các flow, tính toán feature và gửi lên API"""
        while self.running:
            time.sleep(CAPTURE_WINDOW)
            
            current_flows = []
            with self.lock:
                # Copy và clear flows cũ để bắt đầu chu kỳ mới
                current_flows = list(self.active_flows.items())
                self.active_flows.clear()

            if not current_flows:
                continue

            logger.info(f"Analyzing {len(current_flows)} captured flows...")
            
            for key, flow_stats in current_flows:
                features = flow_stats.get_features()
                if not features: continue

                # Chỉ gửi các flow có dữ liệu đáng ngờ hoặc tất cả (tùy chỉnh)
                # Ở đây ta gửi tất cả để test
                try:
                    # Gửi lên API Django
                    response = requests.post(API_URL, json={'features': features}, timeout=1)
                    if response.status_code == 200:
                        result = response.json()
                        pred = result.get('prediction', 'Unknown')
                        conf = result.get('confidence', 0)
                        
                        # In log màu mè chút cho dễ nhìn
                        if pred == 'Attack Detected':
                            logger.warning(f"🚨 ALERT: {pred} | IP: {key[0]}->{key[1]} | Conf: {conf:.2f}")
                        else:
                            logger.info(f"✅ Normal: {key[0]}->{key[1]}")
                            
                except Exception as e:
                    logger.error(f"Failed to send to API: {e}")

    def start(self):
        logger.info(f"Starting Traffic Monitor on interface {INTERFACE}...")
        
        # Chạy luồng phân tích nền
        analyzer_thread = Thread(target=self.analyzer_loop)
        analyzer_thread.daemon = True
        analyzer_thread.start()

        # Bắt đầu bắt gói tin (Block main thread)
        # filter="ip" để bắt gói IP, prn là hàm callback
        sniff(iface=INTERFACE, prn=self.packet_callback, store=0)

if __name__ == "__main__":
    monitor = TrafficMonitor()
    try:
        monitor.start()
    except KeyboardInterrupt:
        logger.info("Stopping monitor...")
