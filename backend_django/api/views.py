import os
import pickle
import logging
import json
import random
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.conf import settings

# --- KHU VỰC SỬA LỖI ---
# Bạn phải import cả ensure_csrf_cookie VÀ csrf_exempt
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
# -----------------------

from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and scaler
MODEL_PATH = os.path.join(settings.BASE_DIR, 'api', 'ml_models', 'random_forest_model.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'api', 'ml_models', 'scaler.pkl')
FEATURE_LIST_PATH = os.path.join(settings.BASE_DIR, 'api', 'ml_models', 'feature_list.pkl')

def generate_random_features():
    """Generate random values for traffic features."""
    return {
        'Bwd Packet Length Min': random.uniform(0, 1000),
        'Bwd Packet Length Std': random.uniform(0, 5000),
        'Init Win bytes forward': random.uniform(0, 65535),
        'Init Win bytes backward': random.uniform(0, 65535),
        'Fwd PSH Flags': random.uniform(0, 1),
        'Average Packet Size': random.uniform(0, 5000),
        'PSH Flag Count': random.uniform(0, 100),
        'Bwd Packets/s': random.uniform(0, 100000),
        'Bwd Header Length': random.uniform(0, 5000),
        'Bwd Packet Length Mean': random.uniform(0, 5000),
        'Fwd Header Length': random.uniform(0, 5000),
        'Packet Length Std': random.uniform(0, 5000),
        'Packet Length Mean': random.uniform(0, 5000),
        'Avg Bwd Segment Size': random.uniform(0, 5000),
        'Fwd Header Length 1': random.uniform(0, 5000),
        'min seg size forward': random.uniform(0, 1000),
        'Bwd Packet Length Max': random.uniform(0, 10000),
        'Packet Length Variance': random.uniform(0, 1000000),
        'Fwd Packet Length Max': random.uniform(0, 10000),
        'Fwd Packet Length Mean': random.uniform(0, 5000),
        'Flow Bytes/s': random.uniform(0, 10000000),
        'Fwd IAT Min': random.uniform(0, 0.001),
        'Total Length of Fwd Packets': random.uniform(0, 1000000),
        'Flow IAT Mean': random.uniform(0, 0.001)
    }

# Load Model Logic
try:
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(FEATURE_LIST_PATH, 'rb') as f:
        feature_list = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    scaler = None
    feature_list = None

# Decorator này dùng để đảm bảo cookie CSRF được gửi về client (Browser)
@ensure_csrf_cookie
def traffic_analysis_form(request):
    """Render the traffic analysis form."""
    return render(request, 'detection_form.html')

# Decorator csrf_exempt này cho phép API nhận request từ Python script mà không cần token
@method_decorator(csrf_exempt, name='dispatch')
class TrafficAnalysisAPIView(View):
    """API view for traffic analysis predictions."""
    
    def get(self, request):
        """Handle GET requests to set CSRF cookie."""
        return JsonResponse({'status': 'ok'})
    
    def post(self, request):
        """Handle POST requests for traffic analysis."""
        if model is None or scaler is None or feature_list is None:
            return JsonResponse({'error': 'Model not loaded'}, status=500)

        try:
            data = json.loads(request.body)
            # Ưu tiên lấy features từ request gửi lên
            features = data.get('features', generate_random_features())
            
            # Prepare input data
            input_data = np.array([features[feature] for feature in feature_list]).reshape(1, -1)
            scaled_data = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(scaled_data)[0]
            probabilities = model.predict_proba(scaled_data)[0]
            confidence = max(probabilities)
            
            result = {
                'prediction': 'Attack Detected' if prediction == 1 else 'Normal Traffic',
                'confidence': float(confidence),
                'probabilities': {
                    'normal': float(probabilities[0]),
                    'attack': float(probabilities[1])
                },
                'input_values': features
            }
            
            logger.info(f"Result: {result['prediction']}")
            return JsonResponse(result)
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
