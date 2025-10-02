import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ParkinsonsPredictor:
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.scaler = None
        
    def load_and_explore_data(self, file_path):
        """Load and perform initial exploration of the dataset"""
        print("📊 Loading and exploring the dataset...")
        
        # Load the dataset
        self.data = pd.read_csv(file_path)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"\nColumn names: {list(self.data.columns)}")
        print(f"\nFirst few rows:")
        print(self.data.head())
        
        # Check for missing values
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        
        # Basic statistics
        print(f"\nBasic statistics:")
        print(self.data.describe())
        
        # Check target variable distribution
        target_col = 'status' if 'status' in self.data.columns else self.data.columns[-1]
        print(f"\nTarget variable '{target_col}' distribution:")
        print(self.data[target_col].value_counts())
        
        return self.data
    
    def visualize_data(self):
        """Create visualizations to understand the data better"""
        print("📈 Creating data visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # Target variable distribution
        target_col = 'status' if 'status' in self.data.columns else self.data.columns[-1]
        
        plt.subplot(3, 4, 1)
        self.data[target_col].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('Target Variable Distribution')
        plt.ylabel('Count')
        
        # Correlation heatmap (top features)
        plt.subplot(3, 4, 2)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        top_corr_features = correlation_matrix[target_col].abs().sort_values(ascending=False)[1:11]
        heatmap_features = list(top_corr_features.index) + [target_col]
        sns.heatmap(self.data[heatmap_features].corr(), 
                   annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Top 10 Features Correlation with Target')
        
        # Feature distribution plots for top correlated features
        top_features = top_corr_features.head(6).index
        for i, feature in enumerate(top_features, 3):
            plt.subplot(3, 4, i)
            for status in self.data[target_col].unique():
                subset = self.data[self.data[target_col] == status]
                plt.hist(subset[feature], alpha=0.7, 
                        label=f'Status {status}', bins=20)
            plt.title(f'{feature} Distribution')
            plt.legend()
            plt.xlabel(feature)
            plt.ylabel('Frequency')
        
        # Box plots for top features
        for i, feature in enumerate(top_features[:3], 9):
            plt.subplot(3, 4, i)
            self.data.boxplot(column=feature, by=target_col, ax=plt.gca())
            plt.title(f'{feature} by Status')
            plt.suptitle('')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess the data for machine learning"""
        print("🔧 Preprocessing the data...")
        
        # Identify target column
        target_col = 'status' if 'status' in self.data.columns else self.data.columns[-1]
        
        # Separate features and target
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle any categorical features if present
        X = pd.get_dummies(X, drop_first=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        print(f"Target distribution in training set:\n{pd.Series(y_train).value_counts()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def feature_selection(self, X_train, y_train, feature_names, k=20):
        """Perform optimized feature selection to identify most important features"""
        print(f"🎯 Performing feature selection on {len(feature_names)} features...")
        
        # Method 1: Fast statistical feature selection (univariate)
        print("   - Running statistical feature selection...")
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        selected_features_statistical = [feature_names[i] for i in selector.get_support(indices=True)]
        
        # Method 2: Random Forest feature importance (optimized)
        print("   - Training Random Forest for feature importance...")
        rf = RandomForestClassifier(
            n_estimators=50,  # Reduced from 100 for speed
            max_depth=10,     # Limit depth for speed
            random_state=42,
            n_jobs=-1         # Use all cores
        )
        rf.fit(X_train, y_train)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        selected_features_rf = feature_importance.head(k)['feature'].tolist()
        
        # Skip RFE as it's too slow - use correlation-based selection instead
        print("   - Computing correlation-based selection...")
        # Calculate correlation with target
        correlations = []
        for i, feature in enumerate(feature_names):
            corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        corr_importance = pd.DataFrame({
            'feature': feature_names,
            'correlation': correlations
        }).sort_values('correlation', ascending=False)
        
        selected_features_corr = corr_importance.head(k)['feature'].tolist()
        
        print(f"\nTop {k} features by statistical method (F-test):")
        print(f"   {selected_features_statistical[:5]}... (showing first 5)")
        print(f"\nTop {k} features by Random Forest importance:")
        print(f"   {selected_features_rf[:5]}... (showing first 5)")
        print(f"\nTop {k} features by correlation:")
        print(f"   {selected_features_corr[:5]}... (showing first 5)")
        
        # Combine methods: use RF importance as primary, but include high correlation features
        final_features = list(dict.fromkeys(selected_features_rf + selected_features_corr))[:k]
        feature_names_list = list(feature_names)
        selected_indices = [feature_names_list.index(feat) for feat in final_features]
        
        # Visualize top features
        plt.figure(figsize=(12, 8))
        top_importance = feature_importance.head(k)
        plt.barh(range(len(top_importance)), top_importance['importance'], color='skyblue')
        plt.yticks(range(len(top_importance)), 
                  [f[:30] + '...' if len(f) > 30 else f for f in top_importance['feature']])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {k} Most Important Features (Random Forest)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        print(f"\n✅ Selected {len(final_features)} features for training")
        
        return selected_indices, final_features
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple ML models and compare their performance"""
        print("🤖 Training multiple machine learning models...")
        
        # Define models with their hyperparameters for tuning
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'Neural Network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
        }
        
        # Train and evaluate models
        results = {}
        trained_models = {}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, config in models_config.items():
            print(f"\n🔍 Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=cv, 
                scoring='accuracy', 
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='accuracy')
            
            results[name] = {
                'model': best_model,
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_params': grid_search.best_params_,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            trained_models[name] = best_model
            
            print(f"✅ {name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        self.models = trained_models
        return results
    
    def evaluate_models(self, results, y_test):
        """Comprehensive evaluation of all trained models"""
        print("\n📊 Model Evaluation Results:")
        print("=" * 80)
        
        # Create results DataFrame
        eval_df = pd.DataFrame({
            name: [res['accuracy'], res['auc_score'], res['cv_mean'], res['cv_std']]
            for name, res in results.items()
        }, index=['Test Accuracy', 'AUC Score', 'CV Mean', 'CV Std']).T
        
        eval_df = eval_df.sort_values('Test Accuracy', ascending=False)
        print(eval_df.round(4))
        
        # Find best model
        best_model_name = eval_df.index[0]
        self.best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"Best Parameters: {results[best_model_name]['best_params']}")
        
        # Detailed evaluation for best model
        print(f"\n📋 Detailed Classification Report for {best_model_name}:")
        print(classification_report(y_test, results[best_model_name]['y_pred']))
        
        # Confusion Matrix
        plt.figure(figsize=(15, 12))
        
        # Confusion matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # ROC Curves
        plt.subplot(2, 3, 2)
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC: {res['auc_score']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        
        # Model comparison
        plt.subplot(2, 3, 3)
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        bars = plt.bar(range(len(model_names)), accuracies, color=colors)
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # AUC comparison
        plt.subplot(2, 3, 4)
        auc_scores = [results[name]['auc_score'] for name in model_names]
        bars = plt.bar(range(len(model_names)), auc_scores, color=colors)
        plt.xlabel('Models')
        plt.ylabel('AUC Score')
        plt.title('Model AUC Score Comparison')
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        
        for bar, auc in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{auc:.3f}', ha='center', va='bottom')
        
        # CV scores with error bars
        plt.subplot(2, 3, 5)
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        plt.bar(range(len(model_names)), cv_means, yerr=cv_stds, 
               capsize=5, color=colors, alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Cross-Validation Scores with Std Dev')
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        
        # Feature importance for best model (if available)
        plt.subplot(2, 3, 6)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            plt.bar(range(len(indices)), importances[indices])
            plt.xlabel('Feature Index')
            plt.ylabel('Importance')
            plt.title(f'Top 10 Feature Importances - {best_model_name}')
        else:
            plt.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
        return best_model_name, self.best_model
    
    def create_ensemble_model(self, results, X_test, y_test):
        """Create an ensemble model combining top performers"""
        print("\n🎭 Creating Ensemble Model...")
        
        # Select top 3 models based on accuracy
        top_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:3]
        
        ensemble_models = [(name, res['model']) for name, res in top_models]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft'  # Use predicted probabilities
        )
        
        # We need to refit on training data, but for demo we'll use the individual predictions
        # In practice, you'd fit the ensemble on the training data
        
        # Simple ensemble prediction (average of probabilities)
        ensemble_proba = np.mean([res['y_pred_proba'] for _, res in top_models], axis=0)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        
        print(f"🎯 Ensemble Model Results:")
        print(f"   Models used: {[name for name, _ in top_models]}")
        print(f"   Accuracy: {ensemble_accuracy:.4f}")
        print(f"   AUC Score: {ensemble_auc:.4f}")
        
        return ensemble, ensemble_accuracy, ensemble_auc
    
    def predict_new_sample(self, features):
        """Make prediction on new sample"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        # Scale the features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = self.best_model.predict(features_scaled)[0]
        probability = self.best_model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    
    def run_complete_pipeline(self, file_path):
        """Run the complete ML pipeline"""
        print("🚀 Starting Complete Parkinson's Disease Prediction Pipeline")
        print("=" * 70)
        
        # Load and explore data
        self.load_and_explore_data(file_path)
        
        # Visualize data
        self.visualize_data()
        
        # Preprocess data
        X_train, X_test, y_train, y_test, feature_names = self.preprocess_data()
        
        # Feature selection
        selected_indices, selected_features = self.feature_selection(
            X_train, y_train, feature_names, k=15
        )
        
        # Apply feature selection
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        # Train models
        results = self.train_models(X_train_selected, X_test_selected, y_train, y_test)
        
        # Evaluate models
        best_model_name, best_model = self.evaluate_models(results, y_test)
        
        # Create ensemble
        ensemble, ensemble_acc, ensemble_auc = self.create_ensemble_model(
            results, X_test_selected, y_test
        )
        
        print("\n" + "=" * 70)
        print("🎉 Pipeline Complete!")
        print(f"🏆 Best Individual Model: {best_model_name}")
        print(f"📊 Best Accuracy: {max([res['accuracy'] for res in results.values()]):.4f}")
        print(f"🎭 Ensemble Accuracy: {ensemble_acc:.4f}")
        print("=" * 70)
        
        return results, best_model, ensemble

# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = ParkinsonsPredictor()
    
    # Run the complete pipeline
    # Replace 'parkinsons_data.csv' with your actual file path
    try:
        results, best_model, ensemble = predictor.run_complete_pipeline('pd_speech_features.csv')
        
        # Example prediction on new data
        # Replace with actual feature values
        sample_features = [0.1, 0.2, 0.3]  # Replace with actual feature values
        # prediction, probability = predictor.predict_new_sample(sample_features)
        # print(f"\nPrediction for new sample: {prediction}")
        # print(f"Probability: {probability}")
        
    except FileNotFoundError:
        print("⚠️  Please make sure to:")
        print("1. Download the dataset from the Kaggle link")
        print("2. Update the file path in the code")
        print("3. Ensure all required libraries are installed:")
        print("   pip install pandas numpy matplotlib seaborn scikit-learn xgboost")