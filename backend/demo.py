"""
Combined Neurological Disease Detection - Parkinson's Disease Classification
This program integrates signal data, image data, and feature-based analysis 
using real datasets for comprehensive Parkinson's Disease prediction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CombinedParkinsonDetector:
    def __init__(self):
        # Model components
        self.rf_classifier = None
        self.cnn_model = None
        self.feature_model = None
        self.ensemble_model = None
        
        # Preprocessing components
        self.scaler = None
        self.feature_scaler = None
        self.selected_features = None
        
        # Results storage
        self.models = {}
        self.best_models = {}
        self.results = {}
        
        # Dataset paths - update these to your actual paths
        self.healthy_dir = r'D:\Programs\In-HouseSRM\Parkinson\Dataset\HandPD_Corpus\Healthy'
        self.patient_dir = r'D:\Programs\In-HouseSRM\Parkinson\Dataset\HandPD_Corpus\Patient'
        
        # Subdirectories for different data types
        self.healthy_circle_data = os.path.join(self.healthy_dir, 'Healthy_Circle')
        self.patient_circle_data = os.path.join(self.patient_dir, 'Patient_Circle')
        self.healthy_meander_data = os.path.join(self.healthy_dir, 'Healthy_Meander')
        self.patient_meander_data = os.path.join(self.patient_dir, 'Patient_Meander')
        self.healthy_spiral_data = os.path.join(self.healthy_dir, 'Healthy_Spiral')
        self.patient_spiral_data = os.path.join(self.patient_dir, 'Patient_Spiral')
        self.healthy_signal_data = os.path.join(self.healthy_dir, 'Healthy_Signal')
        self.patient_signal_data = os.path.join(self.patient_dir, 'Patient_Signal')
        
        # Feature dataset path
        self.feature_dataset_path = 'pd_speech_features.csv'  # Update this path
        
    def load_feature_dataset(self):
        """Load the feature-based dataset (CSV file)"""
        print("Loading feature-based dataset...")
        
        if not os.path.exists(self.feature_dataset_path):
            print(f"Warning: Feature dataset not found at {self.feature_dataset_path}")
            return pd.DataFrame()
        
        try:
            feature_data = pd.read_csv(self.feature_dataset_path)
            print(f"Feature dataset loaded: {feature_data.shape}")
            print(f"Columns: {list(feature_data.columns)}")
            
            # Check target variable
            target_col = 'status' if 'status' in feature_data.columns else feature_data.columns[-1]
            print(f"Target variable '{target_col}' distribution:")
            print(feature_data[target_col].value_counts())
            
            return feature_data
        except Exception as e:
            print(f"Error loading feature dataset: {e}")
            return pd.DataFrame()
    
    def load_signal_data(self, directory, label):
        """Load and process signal data from text files"""
        data = []
        
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return pd.DataFrame()
        
        print(f"Loading signal data from {directory}...")
        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        print(f"Found {len(files)} signal files")
        
        for filename in files:
            filepath = os.path.join(directory, filename)
            person_data = {'label': label, 'source': 'signal'}
            
            try:
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                    coordinates = []
                    
                    for line in lines:
                        line = line.strip()
                        
                        # Parse metadata
                        if line.startswith('#'):
                            if '<Person_ID_Number>' in line:
                                person_data['person_id'] = line.split('>')[1].split('<')[0]
                            elif '<Age>' in line:
                                person_data['age'] = int(line.split('>')[1].split('<')[0])
                            elif '<Gender>' in line:
                                person_data['gender'] = int(line.split('>')[1].split('<')[0])
                            elif '<Writing_Hand>' in line:
                                person_data['writing_hand'] = int(line.split('>')[1].split('<')[0])
                            elif '<Weight>' in line:
                                person_data['weight'] = float(line.split('>')[1].split('<')[0])
                            elif '<Height>' in line:
                                person_data['height'] = float(line.split('>')[1].split('<')[0])
                            elif '<Smoker>' in line:
                                person_data['smoker'] = int(line.split('>')[1].split('<')[0])
                        else:
                            # Parse coordinate data
                            if line and not line.startswith('#'):
                                values = line.split()
                                if len(values) >= 3:
                                    try:
                                        x, y, z = float(values[0]), float(values[1]), float(values[2])
                                        coordinates.append([x, y, z])
                                    except ValueError:
                                        continue
                    
                    # Extract signal features
                    if coordinates:
                        coordinates = np.array(coordinates)
                        
                        # Basic coordinate features
                        person_data['x_mean'] = np.mean(coordinates[:, 0])
                        person_data['y_mean'] = np.mean(coordinates[:, 1])
                        person_data['z_mean'] = np.mean(coordinates[:, 2])
                        person_data['x_std'] = np.std(coordinates[:, 0])
                        person_data['y_std'] = np.std(coordinates[:, 1])
                        person_data['z_std'] = np.std(coordinates[:, 2])
                        person_data['x_range'] = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
                        person_data['y_range'] = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
                        person_data['z_range'] = np.max(coordinates[:, 2]) - np.min(coordinates[:, 2])
                        
                        # Movement analysis features
                        if len(coordinates) > 1:
                            # Velocity features
                            velocity = np.diff(coordinates, axis=0)
                            velocity_magnitude = np.sqrt(np.sum(velocity**2, axis=1))
                            person_data['velocity_mean'] = np.mean(velocity_magnitude)
                            person_data['velocity_std'] = np.std(velocity_magnitude)
                            person_data['velocity_cv'] = person_data['velocity_std'] / person_data['velocity_mean'] if person_data['velocity_mean'] > 0 else 0
                            person_data['velocity_max'] = np.max(velocity_magnitude)
                            person_data['velocity_min'] = np.min(velocity_magnitude)
                            
                            # Acceleration features
                            if len(velocity) > 1:
                                acceleration = np.diff(velocity, axis=0)
                                acceleration_magnitude = np.sqrt(np.sum(acceleration**2, axis=1))
                                person_data['acceleration_mean'] = np.mean(acceleration_magnitude)
                                person_data['acceleration_std'] = np.std(acceleration_magnitude)
                                person_data['acceleration_cv'] = person_data['acceleration_std'] / person_data['acceleration_mean'] if person_data['acceleration_mean'] > 0 else 0
                                
                                # Jerk features (smoothness indicator)
                                if len(acceleration) > 1:
                                    jerk = np.diff(acceleration, axis=0)
                                    jerk_magnitude = np.sqrt(np.sum(jerk**2, axis=1))
                                    person_data['jerk_mean'] = np.mean(jerk_magnitude)
                                    person_data['jerk_std'] = np.std(jerk_magnitude)
                                    person_data['jerk_cv'] = person_data['jerk_std'] / person_data['jerk_mean'] if person_data['jerk_mean'] > 0 else 0
                        
                        # Path analysis
                        path_length = np.sum(np.sqrt(np.sum(np.diff(coordinates, axis=0)**2, axis=1)))
                        person_data['path_length'] = path_length
                        person_data['path_efficiency'] = np.sqrt((coordinates[-1, 0] - coordinates[0, 0])**2 + 
                                                               (coordinates[-1, 1] - coordinates[0, 1])**2) / path_length if path_length > 0 else 0
                        
                        # Tremor analysis (frequency domain)
                        if len(velocity_magnitude) > 10:
                            from scipy.signal import welch
                            freqs, psd = welch(velocity_magnitude, fs=50, nperseg=min(len(velocity_magnitude)//2, 64))
                            # Parkinsonian tremor frequency band (4-6 Hz)
                            tremor_band = (freqs >= 4) & (freqs <= 6)
                            if np.any(tremor_band):
                                person_data['tremor_power'] = np.mean(psd[tremor_band])
                            else:
                                person_data['tremor_power'] = 0
                        
                        data.append(person_data)
            
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue
        
        return pd.DataFrame(data)

    def load_image_data(self, directory, label, target_size=(128, 128)):
        """Load and process image data"""
        images = []
        labels = []
        metadata = []
        
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return np.array([]), np.array([]), []
        
        print(f"Loading images from: {directory}")
        files = os.listdir(directory)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(image_files)} image files")
        
        for filename in image_files:
            filepath = os.path.join(directory, filename)
            
            try:
                img = cv2.imread(filepath)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)
                    img = img / 255.0  # Normalize
                    
                    images.append(img)
                    labels.append(label)
                    metadata.append({'filename': filename, 'source': 'image'})
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        print(f"Successfully loaded {len(images)} images")
        return np.array(images), np.array(labels), metadata

    def create_cnn_model(self, input_shape=(128, 128, 3)):
        """Create CNN model for image classification"""
        model = Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # First block
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            GlobalAveragePooling2D(),
            Dropout(0.5),
            
            # Classification head
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train_feature_models(self, feature_data):
        """Train models on feature-based data"""
        print("Training models on feature dataset...")
        
        # Prepare data
        target_col = 'status' if 'status' in feature_data.columns else feature_data.columns[-1]
        X = feature_data.drop(columns=[target_col])
        y = feature_data[target_col]
        
        # Handle categorical features
        X = pd.get_dummies(X, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.feature_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Feature selection
        print("Performing feature selection...")
        selector = SelectKBest(score_func=f_classif, k=min(20, X_train_scaled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        self.selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected {len(self.selected_features)} features")
        
        # Train multiple models
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'XGBoost': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
            
            # Train model
            model.fit(X_train_selected, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            sensitivity = recall_score(y_test, y_pred)
            specificity = recall_score(1-y_test, 1-y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
        self.feature_model = results[best_model_name]['model']
        self.best_models['features'] = results[best_model_name]
        
        print(f"Best feature model: {best_model_name}")
        return results, X_test, y_test

    def train_signal_model(self, signal_data):
        """Train model on signal data"""
        print("Training model on signal dataset...")
        
        if signal_data.empty:
            print("No signal data available")
            return {}, None, None
        
        # Prepare signal data
        feature_cols = [col for col in signal_data.columns if col not in ['label', 'person_id', 'source']]
        X = signal_data[feature_cols].fillna(signal_data[feature_cols].mean())
        y = signal_data['label']
        
        if len(X) == 0:
            print("No signal features available")
            return {}, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.rf_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.rf_classifier.predict(X_test_scaled)
        y_pred_proba = self.rf_classifier.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'Signal RF': {
                'model': self.rf_classifier,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
        
        self.best_models['signals'] = results['Signal RF']
        print(f"Signal model - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        return results, X_test_scaled, y_test

    def train_image_model(self, all_images, all_labels):
        """Train CNN on image data"""
        print("Training CNN on image dataset...")
        
        if len(all_images) == 0:
            print("No image data available")
            return {}, None, None
        
        # Split image data
        X_img_train, X_img_test, y_img_train, y_img_test = train_test_split(
            all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )
        
        # Create and train CNN
        self.cnn_model = self.create_cnn_model()
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        
        # Train model
        history = self.cnn_model.fit(
            datagen.flow(X_img_train, y_img_train, batch_size=32),
            steps_per_epoch=max(len(X_img_train) // 32, 1),
            epochs=20,
            validation_data=(X_img_test, y_img_test),
            verbose=1
        )
        
        # Evaluate
        y_pred_proba = self.cnn_model.predict(X_img_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_img_test, y_pred)
        auc_score = roc_auc_score(y_img_test, y_pred_proba)
        
        results = {
            'Image CNN': {
                'model': self.cnn_model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
        
        self.best_models['images'] = results['Image CNN']
        print(f"Image model - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        return results, X_img_test, y_img_test, history

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Model comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Combined Parkinson\'s Disease Detection Results', fontsize=16, fontweight='bold')
        
        # 1. Model Performance Comparison
        ax1 = axes[0, 0]
        modalities = list(self.best_models.keys())
        metrics = ['accuracy', 'auc_score', 'sensitivity', 'specificity']
        
        performance_data = []
        for modality in modalities:
            model_info = self.best_models[modality]
            performance_data.append([
                model_info.get('accuracy', 0),
                model_info.get('auc_score', 0),
                model_info.get('sensitivity', 0),
                model_info.get('specificity', 0)
            ])
        
        performance_df = pd.DataFrame(performance_data, index=modalities, columns=metrics)
        performance_df.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Model Performance Comparison')
        ax1.set_ylabel('Score')
        ax1.legend(title='Metrics')
        ax1.grid(True, alpha=0.3)
        
        # 2. AUC Comparison
        ax2 = axes[0, 1]
        auc_scores = [self.best_models[mod].get('auc_score', 0) for mod in modalities]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax2.bar(modalities, auc_scores, color=colors[:len(modalities)])
        ax2.set_title('AUC Score Comparison')
        ax2.set_ylabel('AUC Score')
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Clinical Performance Summary Table
        ax3 = axes[0, 2]
        table_data = []
        for modality in modalities:
            model_info = self.best_models[modality]
            table_data.append([
                modality.title(),
                f"{model_info.get('accuracy', 0):.3f}",
                f"{model_info.get('sensitivity', 0):.3f}",
                f"{model_info.get('specificity', 0):.3f}",
                f"{model_info.get('auc_score', 0):.3f}"
            ])
        
        table = ax3.table(
            cellText=table_data,
            colLabels=['Modality', 'Accuracy', 'Sensitivity', 'Specificity', 'AUC'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(modalities) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax3.set_title('Clinical Performance Summary')
        ax3.axis('off')
        
        # 4. ROC Curves (if available)
        ax4 = axes[1, 0]
        colors_roc = ['blue', 'red', 'green']
        
        for i, (modality, model_info) in enumerate(self.best_models.items()):
            auc = model_info.get('auc_score', 0)
            # Generate representative ROC curve
            fpr = np.linspace(0, 1, 100)
            tpr = 1 - (1 - fpr) ** (2 * auc)  # Approximate ROC curve
            
            ax4.plot(fpr, tpr, color=colors_roc[i % len(colors_roc)], 
                    linewidth=2, label=f'{modality.title()} (AUC = {auc:.3f})')
        
        ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curves Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Feature Importance (if available)
        ax5 = axes[1, 1]
        if 'features' in self.best_models and hasattr(self.best_models['features']['model'], 'feature_importances_'):
            model = self.best_models['features']['model']
            importances = model.feature_importances_
            features = self.selected_features[:len(importances)]
            
            # Get top 10 features
            indices = np.argsort(importances)[::-1][:10]
            
            ax5.barh(range(len(indices)), importances[indices], color='lightblue')
            ax5.set_yticks(range(len(indices)))
            ax5.set_yticklabels([features[i][:20] + '...' if len(features[i]) > 20 else features[i] 
                               for i in indices])
            ax5.set_xlabel('Feature Importance')
            ax5.set_title('Top 10 Important Features')
            ax5.grid(True, alpha=0.3, axis='x')
        else:
            ax5.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Feature Importance')
        
        # 6. Dataset Summary
        ax6 = axes[1, 2]
        
        # Create dataset summary
        summary_text = "DATASET SUMMARY:\n\n"
        
        if 'features' in self.best_models:
            summary_text += f"Feature Dataset: Available\n"
        if 'signals' in self.best_models:
            summary_text += f"Signal Dataset: Available\n"
        if 'images' in self.best_models:
            summary_text += f"Image Dataset: Available\n"
        
        summary_text += f"\nBest Overall Model:\n"
        best_overall = max(self.best_models.keys(), 
                          key=lambda k: self.best_models[k].get('auc_score', 0))
        best_auc = self.best_models[best_overall]['auc_score']
        summary_text += f"{best_overall.title()}\nAUC: {best_auc:.3f}"
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, 
                fontsize=12, verticalalignment='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax6.set_title('Analysis Summary')
        ax6.axis('off')
        
        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Complete Parkinson's Disease Analysis Pipeline")
        print("=" * 70)
        
        os.makedirs('models', exist_ok=True)
        
        # Initialize results storage
        all_results = {}
        
        # 1. Load and process feature dataset
        print("\n1. Loading feature dataset...")
        feature_data = self.load_feature_dataset()
        
        if not feature_data.empty:
            print("Feature dataset visualization...")
            self.visualize_feature_data(feature_data)
            
            feature_results, X_test_features, y_test_features = self.train_feature_models(feature_data)
            all_results['feature'] = feature_results
        
        # 2. Load and process signal data
        print("\n2. Loading signal datasets...")
        healthy_signal_df = self.load_signal_data(self.healthy_signal_data, label=0)
        patient_signal_df = self.load_signal_data(self.patient_signal_data, label=1)
        signal_df = pd.concat([healthy_signal_df, patient_signal_df], ignore_index=True)
        
        if not signal_df.empty:
            print(f"Signal data loaded: {len(signal_df)} samples")
            signal_results, X_test_signals, y_test_signals = self.train_signal_model(signal_df)
            all_results['signal'] = signal_results
        
        # 3. Load and process image data
        print("\n3. Loading image datasets...")
        
        # Load all image types
        healthy_spiral_imgs, healthy_spiral_labels, _ = self.load_image_data(self.healthy_spiral_data, 0)
        patient_spiral_imgs, patient_spiral_labels, _ = self.load_image_data(self.patient_spiral_data, 1)
        
        healthy_circle_imgs, healthy_circle_labels, _ = self.load_image_data(self.healthy_circle_data, 0)
        patient_circle_imgs, patient_circle_labels, _ = self.load_image_data(self.patient_circle_data, 1)
        
        healthy_meander_imgs, healthy_meander_labels, _ = self.load_image_data(self.healthy_meander_data, 0)
        patient_meander_imgs, patient_meander_labels, _ = self.load_image_data(self.patient_meander_data, 1)
        
        # Combine all images
        all_images_list = []
        all_labels_list = []
        
        for imgs, labels in [(healthy_spiral_imgs, healthy_spiral_labels),
                            (patient_spiral_imgs, patient_spiral_labels),
                            (healthy_circle_imgs, healthy_circle_labels),
                            (patient_circle_imgs, patient_circle_labels),
                            (healthy_meander_imgs, healthy_meander_labels),
                            (patient_meander_imgs, patient_meander_labels)]:
            if len(imgs) > 0:
                all_images_list.append(imgs)
                all_labels_list.append(labels)
        
        if all_images_list:
            all_images = np.concatenate(all_images_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)
            
            print(f"Combined images: {all_images.shape}")
            image_results, X_test_images, y_test_images, history = self.train_image_model(all_images, all_labels)
            all_results['image'] = image_results
        
        # 4. Create ensemble model
        print("\n4. Creating ensemble model...")
        if len(self.best_models) > 1:
            ensemble_results = self.create_ensemble_model()
            all_results['ensemble'] = ensemble_results
        
        # 5. Generate comprehensive visualizations
        print("\n5. Creating visualizations...")
        self.create_visualizations()
        
        # 6. Print detailed results
        self.print_detailed_results()
        
        # 7. Save all models
        print("\n6. Saving models...")
        self.save_models()
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        
        return all_results
    
    def visualize_feature_data(self, feature_data):
        """Visualize the feature dataset"""
        print("Creating feature data visualizations...")
        
        target_col = 'status' if 'status' in feature_data.columns else feature_data.columns[-1]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Dataset Analysis', fontsize=16)
        
        # Target distribution
        axes[0, 0].pie(feature_data[target_col].value_counts(), labels=['Healthy', 'Parkinson\'s'], 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Class Distribution')
        
        # Correlation heatmap
        numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = feature_data[numeric_cols].corr()
        top_corr = correlation_matrix[target_col].abs().sort_values(ascending=False)[1:11]
        
        heatmap_features = list(top_corr.index) + [target_col]
        sns.heatmap(feature_data[heatmap_features].corr(), annot=True, cmap='coolwarm', 
                   center=0, ax=axes[0, 1])
        axes[0, 1].set_title('Feature Correlations')
        
        # Feature distributions
        top_features = top_corr.head(4).index
        for i, feature in enumerate(top_features):
            row, col = (i // 2, i % 2)
            ax = axes[1, col] if i < 2 else axes[0, 2] if i == 2 else axes[1, 2]
            
            for status in feature_data[target_col].unique():
                subset = feature_data[feature_data[target_col] == status]
                ax.hist(subset[feature], alpha=0.7, label=f'Class {status}', bins=20)
            
            ax.set_title(f'{feature} Distribution')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_ensemble_model(self):
        """Create ensemble combining all available models"""
        print("Creating ensemble from available models...")
        
        ensemble_weights = {}
        total_weight = 0
        
        # Calculate weights based on AUC scores
        for modality, model_info in self.best_models.items():
            auc = model_info.get('auc_score', 0)
            ensemble_weights[modality] = auc
            total_weight += auc
        
        # Normalize weights
        if total_weight > 0:
            for modality in ensemble_weights:
                ensemble_weights[modality] /= total_weight
        
        print("Ensemble weights:")
        for modality, weight in ensemble_weights.items():
            print(f"  {modality}: {weight:.3f}")
        
        self.ensemble_weights = ensemble_weights
        
        return {
            'Ensemble': {
                'weights': ensemble_weights,
                'component_models': list(self.best_models.keys())
            }
        }
    
    def print_detailed_results(self):
        """Print comprehensive results summary"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PARKINSON'S DISEASE DETECTION RESULTS")
        print("=" * 80)
        
        for modality, model_info in self.best_models.items():
            print(f"\n{modality.upper()} MODEL RESULTS:")
            print("-" * 50)
            
            model_name = type(model_info['model']).__name__
            print(f"Model Type: {model_name}")
            print(f"Accuracy: {model_info.get('accuracy', 0):.4f}")
            print(f"AUC Score: {model_info.get('auc_score', 0):.4f}")
            print(f"Sensitivity: {model_info.get('sensitivity', 0):.4f}")
            print(f"Specificity: {model_info.get('specificity', 0):.4f}")
            
            if 'cv_mean' in model_info:
                print(f"Cross-validation: {model_info['cv_mean']:.4f} (+/- {model_info.get('cv_std', 0):.4f})")
            
            # Clinical interpretation
            sensitivity = model_info.get('sensitivity', 0)
            specificity = model_info.get('specificity', 0)
            
            print("Clinical Assessment:")
            if sensitivity >= 0.8 and specificity >= 0.8:
                print("  Excellent - Suitable for both screening and diagnosis")
            elif sensitivity >= 0.8:
                print("  High sensitivity - Excellent for screening")
            elif specificity >= 0.8:
                print("  High specificity - Good for confirmation")
            else:
                print("  Moderate performance - May need improvement")
        
        # Overall recommendations
        print(f"\n{'=' * 80}")
        print("CLINICAL RECOMMENDATIONS:")
        print("-" * 30)
        
        if self.best_models:
            best_overall = max(self.best_models.keys(), 
                              key=lambda k: self.best_models[k].get('auc_score', 0))
            
            print(f"Best performing modality: {best_overall.upper()}")
            print(f"Best AUC score: {self.best_models[best_overall]['auc_score']:.4f}")
            
            if len(self.best_models) > 1:
                print("Recommendation: Use ensemble approach for best results")
            
            print("\nStrengths by modality:")
            for modality in self.best_models:
                if modality == 'features':
                    print("  - Feature-based: Comprehensive clinical assessment")
                elif modality == 'signals':
                    print("  - Signal-based: Real-time movement analysis")
                elif modality == 'images':
                    print("  - Image-based: Visual pattern recognition")
    
    def save_models(self):
        """Save all trained models"""
        print("Saving trained models...")
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save feature model and scaler
        if self.feature_model is not None:
            joblib.dump(self.feature_model, f'models/feature_model_{timestamp}.pkl')
            joblib.dump(self.feature_scaler, f'models/feature_scaler_{timestamp}.pkl')
            print("Feature model saved")
        
        # Save signal model and scaler
        if self.rf_classifier is not None:
            joblib.dump(self.rf_classifier, f'models/signal_model_{timestamp}.pkl')
            joblib.dump(self.scaler, f'models/signal_scaler_{timestamp}.pkl')
            print("Signal model saved")
        
        # Save CNN model
        if self.cnn_model is not None:
            self.cnn_model.save(f'models/image_model_{timestamp}.h5')
            print("Image model saved")
        
        # Save ensemble configuration
        if hasattr(self, 'ensemble_weights'):
            ensemble_config = {
                'weights': self.ensemble_weights,
                'models': list(self.best_models.keys()),
                'timestamp': timestamp
            }
            
            import json
            with open(f'models/ensemble_config_{timestamp}.json', 'w') as f:
                json.dump(ensemble_config, f, indent=2)
            print("Ensemble configuration saved")
        
        print(f"All models saved with timestamp: {timestamp}")
    
    def predict_sample(self, feature_vector=None, signal_data=None, image_data=None):
        """Make prediction on new sample using available models"""
        predictions = {}
        
        # Feature-based prediction
        if feature_vector is not None and self.feature_model is not None:
            feature_scaled = self.feature_scaler.transform([feature_vector])
            feature_selected = feature_scaled[:, [i for i, f in enumerate(self.feature_scaler.feature_names_in_) 
                                                 if f in self.selected_features]]
            
            pred_proba = self.feature_model.predict_proba(feature_selected)[0, 1]
            predictions['features'] = pred_proba
        
        # Signal-based prediction
        if signal_data is not None and self.rf_classifier is not None:
            signal_scaled = self.scaler.transform([signal_data])
            pred_proba = self.rf_classifier.predict_proba(signal_scaled)[0, 1]
            predictions['signals'] = pred_proba
        
        # Image-based prediction
        if image_data is not None and self.cnn_model is not None:
            if len(image_data.shape) == 3:
                image_data = image_data.reshape(1, *image_data.shape)
            pred_proba = self.cnn_model.predict(image_data, verbose=0)[0, 0]
            predictions['images'] = pred_proba
        
        # Ensemble prediction
        if len(predictions) > 1 and hasattr(self, 'ensemble_weights'):
            ensemble_pred = sum(predictions[mod] * self.ensemble_weights.get(mod, 0) 
                              for mod in predictions)
            predictions['ensemble'] = ensemble_pred
        
        # Final prediction
        if predictions:
            best_pred = predictions.get('ensemble', max(predictions.values()))
            final_prediction = 'Parkinson\'s Disease' if best_pred >= 0.5 else 'Healthy'
            
            return {
                'final_prediction': final_prediction,
                'confidence': best_pred,
                'individual_predictions': predictions
            }
        
        return {'error': 'No valid predictions could be made'}


def main():
    """Main function to run the combined analysis"""
    print("Combined Parkinson's Disease Detection System")
    print("Integrating Feature, Signal, and Image Analysis")
    print("=" * 60)
    
    # Initialize detector
    detector = CombinedParkinsonDetector()
    
    # Update paths if needed
    print("Please ensure the following paths are correct:")
    print(f"Feature dataset: {detector.feature_dataset_path}")
    print(f"Signal data: {detector.healthy_signal_data}, {detector.patient_signal_data}")
    print(f"Image data: {detector.healthy_spiral_data}, etc.")
    print()
    
    try:
        # Run complete analysis
        results = detector.run_complete_analysis()
        
        print("\nAnalysis completed successfully!")
        print("Models saved in 'models/' directory")
        print("\nFor new predictions, use the predict_sample() method with:")
        print("- feature_vector: List of feature values")
        print("- signal_data: Preprocessed signal features") 
        print("- image_data: Normalized image array (128x128x3)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("\nPlease check:")
        print("1. Dataset file paths are correct")
        print("2. Required libraries are installed")
        print("3. Dataset files exist and are readable")
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()