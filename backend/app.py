"""
Parkinson's Disease Detection Backend API
Supports Drawing Pattern Analysis and Clinical Data Assessment
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import cv2
import os
import warnings
import traceback
import json
import tempfile
from datetime import datetime
import logging
import uuid
import sqlite3
from pathlib import Path
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import time
from collections import defaultdict
import re
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

PORT = int(os.environ.get('PORT', 5000))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('parkinsons_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",  # Local React dev
            "http://localhost:5173",  # Local Vite dev
            "https://parkinsons-disease-detection.vercel.app",  # Your deployed frontend
            "*"  # Allow all (remove in production for security)
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

class Config:
    DATABASE_PATH = os.path.join(BASE_DIR, 'parkinsons_assessments.db')
    REPORTS_DIR = os.path.join(BASE_DIR, 'generated_reports')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    TEMP_DIR = os.path.join(BASE_DIR, 'temp_files')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 3600
    MEDICAL_DISCLAIMER = """
    IMPORTANT MEDICAL DISCLAIMER:
    This screening tool is for educational and research purposes only. 
    It is NOT a substitute for professional medical diagnosis or treatment. 
    Results should NOT be used as the sole basis for medical decisions.
    Consult qualified healthcare professionals for medical advice.
    """

app.config.from_object(Config)

models = {
    'rf_model': None,
    'scaler': None,
    'pca': None,
    'cnn_model': None,
    'drawing_model': None
}

rate_limit_store = defaultdict(list)

def rate_limit(f):
    def decorated_function(*args, **kwargs):
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', 
                                       request.environ.get('REMOTE_ADDR', 'unknown'))
        now = time.time()
        rate_limit_store[client_ip] = [req_time for req_time in rate_limit_store[client_ip] 
                                      if now - req_time < Config.RATE_LIMIT_WINDOW]
        
        if len(rate_limit_store[client_ip]) >= Config.RATE_LIMIT_REQUESTS:
            return jsonify({
                'error': 'Rate limit exceeded. Please try again later.',
                'status': 'error'
            }), 429
        
        rate_limit_store[client_ip].append(now)
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS assessments (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        patient_age INTEGER,
                        patient_gender TEXT,
                        patient_weight REAL,
                        patient_height REAL,
                        patient_writing_hand TEXT,
                        patient_smoker BOOLEAN,
                        drawing_prediction INTEGER,
                        drawing_confidence REAL,
                        drawing_features TEXT,
                        drawing_paths TEXT,
                        clinical_prediction INTEGER,
                        clinical_confidence REAL,
                        clinical_features TEXT,
                        combined_prediction INTEGER,
                        combined_confidence REAL,
                        ensemble_info TEXT,
                        report_path TEXT,
                        ip_address TEXT
                    )
                ''')
                conn.commit()
                logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Database init error: {e}")
            raise

    def save_assessment(self, data):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
            # Add missing columns with default None values
                cursor.execute('''
                    INSERT INTO assessments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('id'), 
                    data.get('timestamp'),
                    data.get('patient_age'), 
                    data.get('patient_gender'),
                    data.get('patient_weight'), 
                    data.get('patient_height'),
                    data.get('patient_writing_hand'), 
                    data.get('patient_smoker'),
                    data.get('drawing_prediction'), 
                    data.get('drawing_confidence'),
                    json.dumps(data.get('drawing_features', {})),
                    json.dumps(data.get('drawing_paths', {})),
                    data.get('clinical_prediction'), 
                    data.get('clinical_confidence'),
                    json.dumps(data.get('clinical_features', {})),
                    data.get('combined_prediction'), 
                    data.get('combined_confidence'),
                    json.dumps(data.get('ensemble_info', {})),
                    data.get('report_path'), 
                    data.get('ip_address'),
                    None,  # Add 6 more None values for missing columns
                    None,
                    None,
                    None,
                    None,
                    None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Save error: {e}")
            traceback.print_exc()
            raise

class ParkinsonPredictor:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load ML models with correct filenames"""  # ← FIXED INDENTATION
        global models
        try:
            os.makedirs(Config.MODELS_DIR, exist_ok=True)
            
            # Load Random Forest model
            rf_paths = [
                os.path.join(Config.MODELS_DIR, 'parkinson_rf_model.pkl'),
                os.path.join(Config.MODELS_DIR, 'rf_model.pkl'),
                os.path.join(Config.MODELS_DIR, 'random_forest_model.pkl')
            ]
            
            for rf_path in rf_paths:
                if os.path.exists(rf_path):
                    try:
                        models['rf_model'] = joblib.load(rf_path)
                        logger.info(f"✓ RF model loaded from: {rf_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load RF from {rf_path}: {e}")
            
            if models['rf_model'] is None:
                logger.warning("⚠ RF model not found. Available files:")
                if os.path.exists(Config.MODELS_DIR):
                    for file in os.listdir(Config.MODELS_DIR):
                        logger.info(f"  - {file}")
            
            # Load Scaler
            scaler_paths = [
                os.path.join(Config.MODELS_DIR, 'scaler.pkl'),
                os.path.join(Config.MODELS_DIR, 'standard_scaler.pkl')
            ]
            
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    try:
                        models['scaler'] = joblib.load(scaler_path)
                        logger.info(f"✓ Scaler loaded from: {scaler_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load scaler from {scaler_path}: {e}")
            
            # Load PCA
            pca_paths = [
                os.path.join(Config.MODELS_DIR, 'pca.pkl'),
                os.path.join(Config.MODELS_DIR, 'pca_model.pkl')
            ]
            
            for pca_path in pca_paths:
                if os.path.exists(pca_path):
                    try:
                        models['pca'] = joblib.load(pca_path)
                        logger.info(f"✓ PCA loaded from: {pca_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load PCA from {pca_path}: {e}")
            
            # Load CNN model - TRY MULTIPLE FORMATS
            cnn_paths = [
                os.path.join(Config.MODELS_DIR, 'parkinson_cnn_model.h5'),
                os.path.join(Config.MODELS_DIR, 'parkinson_cnn_model.keras'),
                os.path.join(Config.MODELS_DIR, 'cnn_model.h5'),
                os.path.join(Config.MODELS_DIR, 'cnn_model.keras'),
                os.path.join(Config.MODELS_DIR, 'model.h5'),
                os.path.join(Config.MODELS_DIR, 'model.keras')
            ]
            
            for cnn_path in cnn_paths:
                if os.path.exists(cnn_path):
                    try:
                        models['cnn_model'] = tf.keras.models.load_model(cnn_path, compile=False)
                        logger.info(f"✓ CNN model loaded from: {cnn_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load CNN from {cnn_path}: {e}")
                        traceback.print_exc()
            
            if models['cnn_model'] is None:
                logger.warning("⚠ CNN model not found or failed to load")
            
            self.models_loaded = True
            
            # Log final status
            logger.info("=" * 50)
            logger.info("MODEL LOADING SUMMARY:")
            logger.info(f"  RF Model: {'✓ LOADED' if models['rf_model'] else '✗ NOT LOADED'}")
            logger.info(f"  Scaler:   {'✓ LOADED' if models['scaler'] else '✗ NOT LOADED'}")
            logger.info(f"  PCA:      {'✓ LOADED' if models['pca'] else '✗ NOT LOADED'}")
            logger.info(f"  CNN Model:{'✓ LOADED' if models['cnn_model'] else '✗ NOT LOADED'}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Critical model loading error: {e}")
            traceback.print_exc()
            self.models_loaded = False
    
    def extract_drawing_features(self, coordinates_dict):
    """
    Extract features matching the RF model training data:
    - RMS (root mean square deviation from template)
    - MAX_BETWEEN_ET_HT (max radius difference)
    - MIN_BETWEEN_ET_HT (min radius difference)
    - STD_DEVIATION_ET_HT (std of radius difference)
    - MRT (mean relative tremor)
    - MAX_HT, MIN_HT, STD_HT (handwritten trace stats)
    - CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT
    """
    features = []
    pattern_quality = {}
    
    try:
        for pattern_name in ['circle', 'spiral', 'meander']:
            coords = coordinates_dict.get(pattern_name, [])
            
            if not coords or len(coords) < 5:
                # Not enough data - append zeros
                features.extend([0] * 8)  # 8 features per pattern
                pattern_quality[pattern_name] = {'completeness': 0, 'smoothness': 0}
                continue
            
            # Convert to numpy array
            if isinstance(coords[0], dict):
                coords_array = np.array([[c['x'], c['y']] for c in coords])
            else:
                coords_array = np.array(coords)
            
            # Calculate center of drawing
            center_x = np.mean(coords_array[:, 0])
            center_y = np.mean(coords_array[:, 1])
            
            # Calculate radius from center for each point (HT - Handwritten Trace)
            ht_radius = np.sqrt((coords_array[:, 0] - center_x)**2 + 
                               (coords_array[:, 1] - center_y)**2)
            
            # Expected radius (ET - Exam Template) based on pattern type
            if pattern_name == 'circle':
                # For circle, expected radius is constant (mean of actual radius)
                et_radius = np.ones_like(ht_radius) * np.mean(ht_radius)
            elif pattern_name == 'spiral':
                # For spiral, radius increases linearly
                et_radius = np.linspace(np.min(ht_radius), np.max(ht_radius), len(ht_radius))
            else:  # meander/wave
                # For meander, radius varies sinusoidally
                angles = np.linspace(0, 4*np.pi, len(ht_radius))
                et_radius = np.mean(ht_radius) + 0.3 * np.mean(ht_radius) * np.sin(angles)
            
            # Calculate differences between ET and HT
            diff_et_ht = et_radius - ht_radius
            
            # 1. RMS (Root Mean Square)
            rms = np.sqrt(np.mean(diff_et_ht**2))
            
            # 2. MAX_BETWEEN_ET_HT
            max_diff = np.max(np.abs(diff_et_ht))
            
            # 3. MIN_BETWEEN_ET_HT
            min_diff = np.min(np.abs(diff_et_ht))
            
            # 4. STD_DEVIATION_ET_HT
            std_diff = np.std(diff_et_ht)
            
            # 5. MRT (Mean Relative Tremor)
            velocity = np.diff(ht_radius)
            mrt = np.mean(np.abs(velocity)) / np.mean(ht_radius) if np.mean(ht_radius) > 0 else 0
            
            # 6. MAX_HT
            max_ht = np.max(ht_radius)
            
            # 7. MIN_HT
            min_ht = np.min(ht_radius)
            
            # 8. STD_HT
            std_ht = np.std(ht_radius)
            
            # 9. CHANGES_FROM_NEGATIVE_TO_POSITIVE
            sign_changes = np.sum(np.diff(np.sign(diff_et_ht)) != 0)
            
            # Append features for this pattern
            pattern_features = [
                float(rms),
                float(max_diff),
                float(min_diff),
                float(std_diff),
                float(mrt),
                float(max_ht),
                float(min_ht),
                float(std_ht),
                float(sign_changes)
            ]
            
            # Replace NaN/Inf with 0
            pattern_features = [f if np.isfinite(f) else 0.0 for f in pattern_features]
            
            features.extend(pattern_features)
            
            # Pattern quality metrics
            completeness = min(1.0, len(coords) / 50)
            smoothness = 1 - min(1.0, rms / max(np.mean(ht_radius), 1))
            
            pattern_quality[pattern_name] = {
                'completeness': float(completeness),
                'smoothness': float(smoothness),
                'rms': float(rms),
                'tremor': float(mrt)
            }
            
            logger.info(f"✓ {pattern_name}: RMS={rms:.2f}, MRT={mrt:.3f}, completeness={completeness:.2f}")
        
        # Total features should be 3 patterns × 9 features = 27
        # Pad or trim to match expected size
        target_size = 27
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        logger.info(f"Final feature vector: {len(features)} features")
        
        return np.array(features).reshape(1, -1), pattern_quality
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        logger.error(traceback.format_exc())
        features = [0.0] * 27
        pattern_quality = {
            'circle': {'completeness': 0, 'smoothness': 0}, 
            'spiral': {'completeness': 0, 'smoothness': 0}, 
            'meander': {'completeness': 0, 'smoothness': 0}
        }
        return np.array(features).reshape(1, -1), pattern_quality
    
    def analyze_drawings(self, coordinates_dict):
        try:
            features, pattern_quality = self.extract_drawing_features(coordinates_dict)
            
            prediction = 0
            confidence = 0.5
            
            if models.get('cnn_model'):
                try:
                    image = self.create_drawing_image(coordinates_dict)
                    image_input = image.reshape(1, 128, 128, 3)
                    pred_prob = models['cnn_model'].predict(image_input, verbose=0)[0][0]
                    prediction = int(pred_prob > 0.5)
                    confidence = pred_prob if prediction else (1 - pred_prob)
                except:
                    pass
            
            if confidence == 0.5:
                prediction, confidence = self.heuristic_drawing_analysis(features[0], pattern_quality)
            
            patterns_completed = sum(1 for v in pattern_quality.values() if v['completeness'] > 0)
            
            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'patterns_completed': patterns_completed,
                'pattern_quality': pattern_quality,
                'features': {
                    'tremor_severity': min(1.0, features[0][16] if len(features[0]) > 16 else 0.2),
                    'movement_fluidity': max(0.0, 1 - (features[0][18] if len(features[0]) > 18 else 0.3)),
                    'pattern_accuracy': np.mean([v['completeness'] for v in pattern_quality.values()])
                }
            }
        except Exception as e:
            logger.error(f"Drawing analysis error: {e}")
            return {'prediction': 0, 'confidence': 0.5, 'error': str(e)}
    
    def heuristic_drawing_analysis(self, features, pattern_quality):
    """Enhanced heuristic using RF-style features"""
    risk_indicators = []
    
    # Extract RMS values for each pattern (indices 0, 9, 18)
    circle_rms = features[0] if len(features) > 0 else 0
    spiral_rms = features[9] if len(features) > 9 else 0
    meander_rms = features[18] if len(features) > 18 else 0
    
    # High RMS indicates poor template matching (Parkinson's symptom)
    avg_rms = np.mean([circle_rms, spiral_rms, meander_rms])
    
    if avg_rms > 15:  # High RMS threshold
        risk_indicators.append(0.7)
    elif avg_rms > 8:
        risk_indicators.append(0.4)
    else:
        risk_indicators.append(0.1)
    
    # Check MRT (Mean Relative Tremor) - indices 4, 13, 22
    circle_mrt = features[4] if len(features) > 4 else 0
    spiral_mrt = features[13] if len(features) > 13 else 0
    meander_mrt = features[22] if len(features) > 22 else 0
    
    avg_mrt = np.mean([circle_mrt, spiral_mrt, meander_mrt])
    
    if avg_mrt > 0.15:  # High tremor
        risk_indicators.append(0.6)
    elif avg_mrt > 0.08:
        risk_indicators.append(0.3)
    
    # Check pattern completeness
    avg_completeness = np.mean([v['completeness'] for v in pattern_quality.values()])
    if avg_completeness < 0.4:
        risk_indicators.append(0.5)
    
    # Calculate risk score
    risk_score = np.mean(risk_indicators) if risk_indicators else 0.2
    
    prediction = 1 if risk_score > 0.45 else 0
    
    # Confidence based on certainty
    distance_from_threshold = abs(risk_score - 0.45)
    confidence = 0.60 + (distance_from_threshold * 1.3)
    confidence = min(0.92, max(0.58, confidence))
    
    logger.info(f"Heuristic: avg_rms={avg_rms:.2f}, avg_mrt={avg_mrt:.3f}, risk={risk_score:.3f}, pred={prediction}, conf={confidence:.3f}")
    
    return prediction, confidence
    
    def create_drawing_image(self, coordinates_dict, size=(128, 128)):
        image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        colors_map = {'circle': (50, 255, 50), 'spiral': (50, 50, 255), 'meander': (255, 50, 50)}
        
        all_coords = []
        for coords in coordinates_dict.values():
            if coords and len(coords) > 1:
                all_coords.extend([[c[0], c[1]] for c in coords])
        
        if not all_coords:
            return image.astype(np.float32) / 255.0
        
        all_coords = np.array(all_coords)
        x_min, x_max = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
        y_min, y_max = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
        
        padding = 10
        scale = min((size[0] - 2*padding) / max(x_max - x_min, 1),
                   (size[1] - 2*padding) / max(y_max - y_min, 1))
        
        for pattern, coords in coordinates_dict.items():
            if coords and len(coords) > 1:
                coords_array = np.array(coords)
                x = ((coords_array[:, 0] - x_min) * scale + padding).astype(int)
                y = ((coords_array[:, 1] - y_min) * scale + padding).astype(int)
                x = np.clip(x, 0, size[0] - 1)
                y = np.clip(y, 0, size[1] - 1)
                
                color = colors_map.get(pattern, (128, 128, 128))
                for i in range(len(x) - 1):
                    cv2.line(image, (x[i], y[i]), (x[i+1], y[i+1]), color, 2)
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    def analyze_clinical(self, patient_data):
        try:
            age = float(patient_data.get('age', 0))
            gender = patient_data.get('gender', 'unknown').lower()
            weight = float(patient_data.get('weight', 70)) if patient_data.get('weight') else 70
            height = float(patient_data.get('height', 170)) if patient_data.get('height') else 170
            smoker = bool(patient_data.get('smoker', False))
            
            features = self.create_clinical_features(age, gender, weight, height, smoker)
            
            prediction = 0
            confidence = 0.5
            
            if all(models.get(m) for m in ['rf_model', 'scaler', 'pca']):
                try:
                    features_scaled = models['scaler'].transform(features)
                    features_pca = models['pca'].transform(features_scaled)
                    prediction = models['rf_model'].predict(features_pca)[0]
                    pred_proba = models['rf_model'].predict_proba(features_pca)[0]
                    confidence = float(np.max(pred_proba))
                except:
                    prediction, confidence = self.evidence_based_assessment(age, gender, smoker)
            else:
                prediction, confidence = self.evidence_based_assessment(age, gender, smoker)
            
            bmi = weight / ((height/100) ** 2) if height > 0 else None
            
            return {
                'prediction': int(prediction),
                'confidence': float(confidence),
                'bmi': bmi,
                'risk_factors': self.analyze_risk_factors(age, gender, bmi, smoker)
            }
        except Exception as e:
            logger.error(f"Clinical analysis error: {e}")
            return {'prediction': 0, 'confidence': 0.5, 'error': str(e)}
    
    def create_clinical_features(self, age, gender, weight, height, smoker, handedness='right'):
    """
    Create features EXACTLY matching RF model training:
    Assuming RF model expects: [age, gender(M=1/F=0), handedness(R=1/L=0)]
    Plus drawing features are added separately
    """
    features = [
        float(age),
        1.0 if gender.lower() == 'male' else 0.0,
        1.0 if handedness.lower() == 'right' else 0.0
    ]
    
    logger.info(f"Clinical features created: age={age}, gender={gender}, handedness={handedness}")
    
    return np.array(features).reshape(1, -1)
    
    def analyze_risk_factors(self, age, gender, bmi, smoker):
        return {
            'age_risk': 'High' if age > 60 else 'Moderate' if age > 50 else 'Low',
            'gender_risk': 'Elevated' if gender == 'male' else 'Standard',
            'bmi_status': 'Normal' if bmi and 18.5 <= bmi < 30 else 'Abnormal' if bmi else 'Unknown',
            'lifestyle': 'Smoker' if smoker else 'Non-smoker'
        }
    
    def create_ensemble(self, drawing_result=None, clinical_result=None):
        try:
            results = []
            weights = []
            
            if drawing_result and drawing_result.get('prediction') is not None:
                results.append(drawing_result)
                patterns = drawing_result.get('patterns_completed', 0)
                weights.append(0.6 * max(0.5, patterns / 3.0))
            
            if clinical_result and clinical_result.get('prediction') is not None:
                results.append(clinical_result)
                weights.append(0.4)
            
            if not results:
                return {'prediction': 0, 'confidence': 0.5, 'ensemble_info': {}}
            
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            predictions = [r['prediction'] for r in results]
            confidences = [r['confidence'] for r in results]
            
            weighted_pred = sum(p * w for p, w in zip(predictions, weights))
            final_pred = 1 if weighted_pred > 0.5 else 0
            
            confidence = np.average(confidences, weights=weights)
            if np.var(predictions) == 0:
                confidence *= 1.1
            
            return {
                'prediction': int(final_pred),
                'confidence': float(min(0.95, confidence)),
                'ensemble_info': {
                    'total_assessments': len(results),
                    'positive_assessments': sum(predictions),
                    'agreement_level': 'Perfect' if np.var(predictions) == 0 else 'Good'
                }
            }
        except Exception as e:
            logger.error(f"Ensemble error: {e}")
            return {'prediction': 0, 'confidence': 0.5, 'error': str(e)}

db_manager = DatabaseManager(Config.DATABASE_PATH)
predictor = ParkinsonPredictor(db_manager)

for directory in [Config.REPORTS_DIR, Config.TEMP_DIR, Config.MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)
# Add this BEFORE the /api/health endpoint in app.py:

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API documentation"""
    return jsonify({
        'message': 'Parkinson\'s Disease Detection API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'analyze_drawings': '/api/analyze-drawings [POST]',
            'analyze_clinical': '/api/analyze-clinical [POST]',
            'analyze_combined': '/api/analyze-combined [POST]',
            'generate_report': '/api/generate-report [POST]',
            'download_report': '/api/download-report/<id> [GET]',
            'export_data': '/api/export-data/<id> [GET]',
            'export_all': '/api/export-all [GET]',
            'stats': '/api/stats [GET]'
        },
        'docs': 'https://github.com/sharvesh24/Parkinsons_Disease_Detection',
        'disclaimer': Config.MEDICAL_DISCLAIMER
    })

@app.route('/api/health', methods=['GET'])
@rate_limit
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'rf_model': models.get('rf_model') is not None,
            'cnn_model': models.get('cnn_model') is not None
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze-drawings', methods=['POST'])
@rate_limit
def analyze_drawings():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400
        
        coordinates = {
            'circle': data.get('circle', []),
            'spiral': data.get('spiral', []),
            'meander': data.get('meander', [])
        }
        
        result = predictor.analyze_drawings(coordinates)
        
        return jsonify({
            'assessment_id': str(uuid.uuid4()),
            'result': result['prediction'],
            'confidence': result['confidence'],
            'features': result.get('features', {}),
            'patterns_completed': result.get('patterns_completed', 0),
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Drawing error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/analyze-clinical', methods=['POST'])
@rate_limit
def analyze_clinical(self, patient_data):
    try:
        age = float(patient_data.get('age', 0))
        gender = patient_data.get('gender', 'unknown').lower()
        weight = float(patient_data.get('weight', 70)) if patient_data.get('weight') else 70
        height = float(patient_data.get('height', 170)) if patient_data.get('height') else 170
        smoker = bool(patient_data.get('smoker', False))
        handedness = patient_data.get('writingHand', 'right')  # ← Use frontend's writingHand
        
        logger.info(f"Clinical data received: age={age}, gender={gender}, handedness={handedness}")
        
        features = self.create_clinical_features(age, gender, weight, height, smoker, handedness)
        
        prediction = 0
        confidence = 0.5
        
        # Try using RF model if loaded
        if all(models.get(m) for m in ['rf_model', 'scaler', 'pca']):
            try:
                logger.info(f"Using RF model with features: {features}")
                features_scaled = models['scaler'].transform(features)
                features_pca = models['pca'].transform(features_scaled)
                prediction = models['rf_model'].predict(features_pca)[0]
                pred_proba = models['rf_model'].predict_proba(features_pca)[0]
                confidence = float(np.max(pred_proba))
                logger.info(f"RF prediction: {prediction}, confidence: {confidence:.3f}")
            except Exception as e:
                logger.error(f"RF model failed: {e}, falling back to heuristic")
                prediction, confidence = self.evidence_based_assessment(age, gender, smoker)
        else:
            logger.warning("RF model not available, using heuristic")
            prediction, confidence = self.evidence_based_assessment(age, gender, smoker)
        
        bmi = weight / ((height/100) ** 2) if height > 0 else None
        
        return {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'bmi': bmi,
            'risk_factors': self.analyze_risk_factors(age, gender, bmi, smoker)
        }
    except Exception as e:
        logger.error(f"Clinical analysis error: {e}")
        traceback.print_exc()
        return {'prediction': 0, 'confidence': 0.5, 'error': str(e)}

@app.route('/api/analyze-combined', methods=['POST'])
@rate_limit
def analyze_combined():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400
        
        drawing = data.get('drawingPrediction')
        clinical = data.get('clinicalPrediction')
        
        ensemble = predictor.create_ensemble(drawing, clinical)
        
        assessment_id = str(uuid.uuid4())
        assessment_data = {
            'id': assessment_id,
            'timestamp': datetime.now().isoformat(),
            'patient_age': clinical.get('age') if clinical else None,
            'patient_gender': clinical.get('gender') if clinical else None,
            'drawing_prediction': drawing.get('result') if drawing else None,
            'drawing_confidence': drawing.get('confidence') if drawing else None,
            'clinical_prediction': clinical.get('result') if clinical else None,
            'clinical_confidence': clinical.get('confidence') if clinical else None,
            'combined_prediction': ensemble['prediction'],
            'combined_confidence': ensemble['confidence'],
            'ensemble_info': ensemble.get('ensemble_info', {}),
            'ip_address': request.environ.get('REMOTE_ADDR', 'unknown')
        }
        
        db_manager.save_assessment(assessment_data)
        
        return jsonify({
            'assessment_id': assessment_id,
            'result': ensemble['prediction'],
            'confidence': ensemble['confidence'],
            'ensemble_info': ensemble.get('ensemble_info', {}),
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Combined error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.create_custom_styles()
    
    def create_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=20
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#0d47a1'),
            spaceBefore=15,
            spaceAfter=10
        ))
        self.styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.red,
            leftIndent=20,
            rightIndent=20
        ))
    
    def generate_report(self, assessment_data, patient_info=None):
        try:
            report_id = assessment_data.get('id', str(uuid.uuid4()))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"parkinsons_report_{report_id}_{timestamp}.pdf"
            filepath = os.path.join(Config.REPORTS_DIR, filename)
            
            doc = SimpleDocTemplate(filepath, pagesize=A4, topMargin=72, bottomMargin=72)
            story = []
            
            # Title
            story.append(Paragraph("Parkinson's Disease Screening Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 20))
            
            # Disclaimer
            story.append(Paragraph("MEDICAL DISCLAIMER", self.styles['SectionHeader']))
            story.append(Paragraph(Config.MEDICAL_DISCLAIMER, self.styles['Disclaimer']))
            story.append(Spacer(1, 20))
            
            # Report Info
            story.append(Paragraph("Report Information", self.styles['SectionHeader']))
            info_data = [
                ['Report ID:', report_id],
                ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Assessment Date:', assessment_data.get('timestamp', 'Unknown')]
            ]
            info_table = Table(info_data, colWidths=[2*inch, 3*inch])
            info_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(info_table)
            story.append(Spacer(1, 15))
            
            # Patient Info
            if patient_info:
                story.append(Paragraph("Patient Information", self.styles['SectionHeader']))
                patient_data = []
                if patient_info.get('age'):
                    patient_data.append(['Age:', f"{patient_info['age']} years"])
                if patient_info.get('gender'):
                    patient_data.append(['Gender:', patient_info['gender'].title()])
                if patient_info.get('weight') and patient_info.get('height'):
                    bmi = float(patient_info['weight']) / ((float(patient_info['height'])/100) ** 2)
                    patient_data.append(['BMI:', f"{bmi:.1f} kg/m²"])
                
                if patient_data:
                    patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
                    patient_table.setStyle(TableStyle([
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ]))
                    story.append(patient_table)
                story.append(Spacer(1, 15))
            
            # Results Summary
            story.append(Paragraph("Assessment Results", self.styles['SectionHeader']))
            combined_pred = assessment_data.get('combined_prediction')
            combined_conf = assessment_data.get('combined_confidence', 0.5)
            result_text = "POSITIVE SCREENING" if combined_pred else "NEGATIVE SCREENING"
            
            story.append(Paragraph(f"""
            <b>Overall Result:</b> {result_text}<br/>
            <b>Confidence:</b> {combined_conf*100:.1f}%<br/><br/>
            This assessment combines drawing pattern analysis and clinical risk factors.
            """, self.styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Drawing Results
            if assessment_data.get('drawing_prediction') is not None:
                story.append(Paragraph("Motor Control Analysis", self.styles['SectionHeader']))
                draw_pred = assessment_data.get('drawing_prediction')
                draw_conf = assessment_data.get('drawing_confidence', 0.5)
                draw_text = "Indicators detected" if draw_pred else "Normal patterns"
                story.append(Paragraph(f"""
                <b>Result:</b> {draw_text}<br/>
                <b>Confidence:</b> {draw_conf*100:.1f}%
                """, self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Clinical Results
            if assessment_data.get('clinical_prediction') is not None:
                story.append(Paragraph("Clinical Risk Assessment", self.styles['SectionHeader']))
                clin_pred = assessment_data.get('clinical_prediction')
                clin_conf = assessment_data.get('clinical_confidence', 0.5)
                clin_text = "Elevated risk" if clin_pred else "Standard risk"
                story.append(Paragraph(f"""
                <b>Result:</b> {clin_text}<br/>
                <b>Confidence:</b> {clin_conf*100:.1f}%
                """, self.styles['Normal']))
                story.append(Spacer(1, 15))
            
            # Recommendations
            story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
            if combined_pred:
                recs = [
                    "1. Consult with a neurologist for comprehensive evaluation",
                    "2. Schedule appointment with primary care physician",
                    "3. Keep detailed symptom diary",
                    "4. Maintain regular physical activity",
                    "5. Follow-up screening in 6-12 months"
                ]
            else:
                recs = [
                    "1. Continue regular health monitoring",
                    "2. Maintain active lifestyle",
                    "3. Follow up with primary care as needed",
                    "4. Consider repeat screening in 1-2 years",
                    "5. Report any new symptoms to healthcare provider"
                ]
            
            for rec in recs:
                story.append(Paragraph(rec, self.styles['Normal']))
            
            doc.build(story)
            logger.info(f"Report generated: {filepath}")
            
            return {'success': True, 'filepath': filepath, 'filename': filename, 'report_id': report_id}
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

report_generator = ReportGenerator()

@app.route('/api/generate-report', methods=['POST'])
@rate_limit
def generate_report():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400
        
        assessment_id = data.get('assessmentId') or str(uuid.uuid4())
        patient_info = data.get('patientInfo', {})
        predictions = data.get('predictions', {})
        
        assessment_data = {
            'id': assessment_id,
            'timestamp': datetime.now().isoformat(),
            'drawing_prediction': predictions.get('drawing', {}).get('result'),
            'drawing_confidence': predictions.get('drawing', {}).get('confidence'),
            'clinical_prediction': predictions.get('clinical', {}).get('result'),
            'clinical_confidence': predictions.get('clinical', {}).get('confidence'),
            'combined_prediction': predictions.get('combined', {}).get('result'),
            'combined_confidence': predictions.get('combined', {}).get('confidence')
        }
        
        result = report_generator.generate_report(assessment_data, patient_info)
        
        if not result['success']:
            return jsonify({'error': result['error'], 'status': 'error'}), 500
        
        return jsonify({
            'report_id': result['report_id'],
            'filename': result['filename'],
            'status': 'success',
            'message': 'Report generated successfully'
        })
    except Exception as e:
        logger.error(f"Report endpoint error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/download-report/<report_id>', methods=['GET'])
@rate_limit
def download_report(report_id):
    try:
        if not re.match(r'^[a-zA-Z0-9_-]+$', report_id):
            return jsonify({'error': 'Invalid report ID', 'status': 'error'}), 400
        
        report_files = [f for f in os.listdir(Config.REPORTS_DIR) 
                       if f.startswith(f'parkinsons_report_{report_id}')]
        
        if not report_files:
            return jsonify({'error': 'Report not found', 'status': 'error'}), 404
        
        report_file = sorted(report_files)[-1]
        filepath = os.path.join(Config.REPORTS_DIR, report_file)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Report file not found', 'status': 'error'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=report_file, 
                        mimetype='application/pdf')
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/export-data/<assessment_id>', methods=['GET'])
@rate_limit
def export_data(assessment_id):
    try:
        if not re.match(r'^[a-zA-Z0-9-]+$', assessment_id):
            return jsonify({'error': 'Invalid assessment ID', 'status': 'error'}), 400
        
        with sqlite3.connect(Config.DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM assessments WHERE id = ?', (assessment_id,))
            result = cursor.fetchone()
            
            if not result:
                return jsonify({'error': 'Assessment not found', 'status': 'error'}), 404
            
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, result))
            
            # Remove sensitive info
            data.pop('ip_address', None)
            
            # Parse JSON fields
            for field in ['drawing_features', 'drawing_paths', 'clinical_features', 'ensemble_info']:
                if data.get(field):
                    try:
                        data[field] = json.loads(data[field])
                    except:
                        pass
        
        return jsonify({
            'assessment_data': data,
            'status': 'success',
            'exported_at': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/export-all', methods=['GET'])
@rate_limit
def export_all():
    try:
        with sqlite3.connect(Config.DATABASE_PATH) as conn:
            df = pd.read_sql_query('SELECT * FROM assessments', conn)
            
            # Remove sensitive columns
            df = df.drop(columns=['ip_address'], errors='ignore')
            
            # Create CSV in temp directory
            export_file = os.path.join(Config.TEMP_DIR, f'assessments_export_{int(time.time())}.csv')
            df.to_csv(export_file, index=False)
            
            return send_file(export_file, as_attachment=True, 
                           download_name='parkinsons_assessments.csv',
                           mimetype='text/csv')
    except Exception as e:
        logger.error(f"Export all error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/stats', methods=['GET'])
@rate_limit
def get_stats():
    try:
        with sqlite3.connect(Config.DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM assessments')
            total = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN drawing_prediction IS NOT NULL THEN 1 ELSE 0 END) as drawing,
                    SUM(CASE WHEN clinical_prediction IS NOT NULL THEN 1 ELSE 0 END) as clinical,
                    SUM(CASE WHEN combined_prediction IS NOT NULL THEN 1 ELSE 0 END) as combined
                FROM assessments
            ''')
            counts = cursor.fetchone()
            
            cursor.execute("SELECT COUNT(*) FROM assessments WHERE timestamp > datetime('now', '-1 day')")
            recent = cursor.fetchone()[0]
        
        return jsonify({
            'total_assessments': total,
            'drawing_assessments': counts[0],
            'clinical_assessments': counts[1],
            'combined_assessments': counts[2],
            'recent_24h': recent,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found', 'status': 'error'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

if __name__ == '__main__':
    logger.info("Starting Parkinson's Detection API...")
    logger.info("Endpoints: /api/health, /api/analyze-drawings, /api/analyze-clinical")
    logger.info("          /api/analyze-combined, /api/generate-report")
    logger.info("          /api/download-report/<id>, /api/export-data/<id>")
    logger.info("          /api/export-all, /api/stats")
    logger.info("MEDICAL DISCLAIMER: For research purposes only")
    app.run(debug=False, host='0.0.0.0', port=PORT, threaded=True)
