"""
Complete CERT r6.2 Behavioral Analytics for Insider Threat Detection
Author: Research Project Implementation
Purpose: End-to-end pipeline for behavioral analytics in remote work environments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
import lightgbm as lgb

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("All libraries imported successfully!")

class CERTDataProcessor:
    """
    Class to handle CERT r6.2 dataset processing and feature extraction
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.logon_data = None
        self.file_data = None
        self.email_data = None
        self.http_data = None
        self.device_data = None
        self.psychometric_data = None
        self.ldap_data = None
        self.processed_features = None
        
    def load_data(self):
        """Load all CERT r6.2 CSV files"""
        print("Loading CERT r6.2 dataset...")
        
        try:
            # Core behavioral data files
            self.logon_data = pd.read_csv(f"{self.data_path}/logon.csv")
            print(f"Logon data loaded: {self.logon_data.shape}")
            
            self.file_data = pd.read_csv(f"{self.data_path}/file.csv")
            print(f"File data loaded: {self.file_data.shape}")
            
            self.email_data = pd.read_csv(f"{self.data_path}/email.csv")
            print(f"Email data loaded: {self.email_data.shape}")
            
            self.http_data = pd.read_csv(f"{self.data_path}/http.csv")
            print(f"HTTP data loaded: {self.http_data.shape}")
            
            self.device_data = pd.read_csv(f"{self.data_path}/device.csv")
            print(f"Device data loaded: {self.device_data.shape}")
            
            # Additional data files
            try:
                self.psychometric_data = pd.read_csv(f"{self.data_path}/psychometric.csv")
                print(f"Psychometric data loaded: {self.psychometric_data.shape}")
            except:
                print("Psychometric data not found or not accessible")
                
            try:
                self.ldap_data = pd.read_csv(f"{self.data_path}/LDAP.csv")
                print(f"LDAP data loaded: {self.ldap_data.shape}")
            except:
                print("LDAP data not found or not accessible")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def explore_data(self):
        """Explore the dataset structure and basic statistics"""
        print("\n=== CERT r6.2 Data Exploration ===")
        
        # Logon data exploration
        if self.logon_data is not None:
            print("\n--- Logon Data ---")
            print(f"Columns: {list(self.logon_data.columns)}")
            print(f"Date range: {self.logon_data['date'].min()} to {self.logon_data['date'].max()}")
            print(f"Unique users: {self.logon_data['user'].nunique()}")
            print(f"Total logon events: {len(self.logon_data)}")
            
        # File data exploration  
        if self.file_data is not None:
            print("\n--- File Data ---")
            print(f"Columns: {list(self.file_data.columns)}")
            print(f"Unique users: {self.file_data['user'].nunique()}")
            print(f"Total file events: {len(self.file_data)}")
            
        # Email data exploration
        if self.email_data is not None:
            print("\n--- Email Data ---")
            print(f"Columns: {list(self.email_data.columns)}")
            print(f"Unique users: {self.email_data['user'].nunique()}")
            print(f"Total email events: {len(self.email_data)}")
            
        # Device data exploration
        if self.device_data is not None:
            print("\n--- Device Data ---")
            print(f"Columns: {list(self.device_data.columns)}")
            print(f"Unique users: {self.device_data['user'].nunique()}")
            print(f"Total device events: {len(self.device_data)}")
            
    def preprocess_data(self):
        """Clean and preprocess all data sources"""
        print("\n=== Data Preprocessing ===")
        
        # Convert date columns to datetime
        for df_name, df in [('logon', self.logon_data), ('file', self.file_data), 
                           ('email', self.email_data), ('http', self.http_data),
                           ('device', self.device_data)]:
            if df is not None and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['hour'] = df['date'].dt.hour
                df['day_of_week'] = df['date'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6])
                print(f"{df_name} data preprocessed")
                
    def extract_behavioral_features(self):
        """Extract comprehensive behavioral features for each user"""
        print("\n=== Feature Extraction ===")
        
        features_list = []
        
        # Get all unique users
        all_users = set()
        for df in [self.logon_data, self.file_data, self.email_data, self.device_data]:
            if df is not None:
                all_users.update(df['user'].unique())
        
        print(f"Processing features for {len(all_users)} users...")
        
        for user in all_users:
            user_features = {'user': user}
            
            # Logon-based features
            if self.logon_data is not None:
                user_logon = self.logon_data[self.logon_data['user'] == user]
                
                # Basic logon statistics
                user_features['total_logons'] = len(user_logon)
                user_features['avg_logons_per_day'] = len(user_logon) / max(1, user_logon['date'].dt.date.nunique())
                
                # Time-based patterns (remote work relevant)
                user_features['after_hours_logons'] = len(user_logon[(user_logon['hour'] < 8) | (user_logon['hour'] > 18)])
                user_features['weekend_logons'] = len(user_logon[user_logon['is_weekend']])
                user_features['night_logons'] = len(user_logon[(user_logon['hour'] >= 22) | (user_logon['hour'] <= 6)])
                
                # Session patterns
                if 'activity' in user_logon.columns:
                    logon_events = user_logon[user_logon['activity'] == 'Logon']
                    logoff_events = user_logon[user_logon['activity'] == 'Logoff']
                    user_features['logon_logoff_ratio'] = len(logon_events) / max(1, len(logoff_events))
                
                # PC diversity (simulates remote work from different locations)
                user_features['unique_pcs'] = user_logon['pc'].nunique() if 'pc' in user_logon.columns else 0
                
            # File access features
            if self.file_data is not None:
                user_files = self.file_data[self.file_data['user'] == user]
                
                user_features['total_file_events'] = len(user_files)
                user_features['unique_files'] = user_files['filename'].nunique() if 'filename' in user_files.columns else 0
                user_features['file_copy_events'] = len(user_files[user_files['activity'] == 'File Copy']) if 'activity' in user_files.columns else 0
                
                # Suspicious file activities
                if 'activity' in user_files.columns:
                    user_features['file_delete_events'] = len(user_files[user_files['activity'] == 'File Delete'])
                    user_features['file_write_events'] = len(user_files[user_files['activity'] == 'File Write'])
                    
            # Email behavior features  
            if self.email_data is not None:
                user_emails = self.email_data[self.email_data['user'] == user]
                
                user_features['total_emails'] = len(user_emails)
                user_features['unique_email_recipients'] = user_emails['to'].nunique() if 'to' in user_emails.columns else 0
                
                # External email patterns (potential data exfiltration)
                if 'to' in user_emails.columns:
                    # Assume external emails contain '@' but not company domain
                    external_emails = user_emails[user_emails['to'].str.contains('@', na=False) & 
                                                ~user_emails['to'].str.contains('dtaa.com', na=False)]
                    user_features['external_emails'] = len(external_emails)
                    
            # Device usage features
            if self.device_data is not None:
                user_devices = self.device_data[self.device_data['user'] == user]
                
                user_features['total_device_events'] = len(user_devices)
                user_features['unique_devices'] = user_devices['filename'].nunique() if 'filename' in user_devices.columns else 0
                
                # USB/removable device usage
                if 'activity' in user_devices.columns:
                    user_features['device_connect_events'] = len(user_devices[user_devices['activity'] == 'Connect'])
                    user_features['device_disconnect_events'] = len(user_devices[user_devices['activity'] == 'Disconnect'])
                    
            # HTTP/web browsing features
            if self.http_data is not None:
                user_http = self.http_data[self.http_data['user'] == user]
                
                user_features['total_web_requests'] = len(user_http)
                user_features['unique_websites'] = user_http['url'].nunique() if 'url' in user_http.columns else 0
                
            features_list.append(user_features)
            
        # Convert to DataFrame
        self.processed_features = pd.DataFrame(features_list)
        
        # Fill missing values
        self.processed_features = self.processed_features.fillna(0)
        
        print(f"Feature extraction completed: {self.processed_features.shape}")
        print(f"Features extracted: {list(self.processed_features.columns)}")
        
        return self.processed_features
    
    def identify_insider_threats(self):
        """Identify known insider threat cases from the dataset"""
        print("\n=== Identifying Insider Threat Cases ===")
        
        # CERT r6.2 typically includes specific insider threat scenarios
        # This is a simplified approach - in reality, you'd need the insider threat documentation
        
        # Create labels based on anomalous behavior patterns
        features_df = self.processed_features.copy()
        
        # Calculate anomaly scores based on multiple factors
        features_df['anomaly_score'] = 0
        
        # High after-hours activity
        if 'after_hours_logons' in features_df.columns:
            features_df['anomaly_score'] += (features_df['after_hours_logons'] > 
                                           features_df['after_hours_logons'].quantile(0.95)).astype(int) * 2
        
        # High file copy activity
        if 'file_copy_events' in features_df.columns:
            features_df['anomaly_score'] += (features_df['file_copy_events'] > 
                                           features_df['file_copy_events'].quantile(0.95)).astype(int) * 3
        
        # High external email activity
        if 'external_emails' in features_df.columns:
            features_df['anomaly_score'] += (features_df['external_emails'] > 
                                           features_df['external_emails'].quantile(0.9)).astype(int) * 2
        
        # High USB device usage
        if 'device_connect_events' in features_df.columns:
            features_df['anomaly_score'] += (features_df['device_connect_events'] > 
                                           features_df['device_connect_events'].quantile(0.9)).astype(int) * 2
        
        # Multiple PC usage (could indicate credential sharing)
        if 'unique_pcs' in features_df.columns:
            features_df['anomaly_score'] += (features_df['unique_pcs'] > 
                                           features_df['unique_pcs'].quantile(0.9)).astype(int) * 1
        
        # Label as insider threat if anomaly score > threshold
        threshold = 3  # Adjust based on your requirements
        features_df['is_insider_threat'] = (features_df['anomaly_score'] >= threshold).astype(int)
        
        insider_count = features_df['is_insider_threat'].sum()
        total_users = len(features_df)
        
        print(f"Identified {insider_count} potential insider threats out of {total_users} users")
        print(f"Insider threat rate: {insider_count/total_users*100:.2f}%")
        
        return features_df

# Behavioral Analytics Models
class BehavioralAnalyticsModels:
    """
    Implementation of various behavioral analytics models for insider threat detection
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def prepare_data(self, features_df):
        """Prepare data for machine learning"""
        print("\n=== Preparing Data for ML ===")
        
        # Separate features and target
        X = features_df.drop(['user', 'is_insider_threat', 'anomaly_score'], axis=1)
        y = features_df['is_insider_threat']
        
        # Handle any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        print(f"Insider threats in training: {y_train.sum()}")
        print(f"Insider threats in test: {y_test.sum()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns
    
    def train_baseline_models(self, X_train, X_test, y_train, y_test):
        """Train baseline models from literature review"""
        print("\n=== Training Baseline Models ===")
        
        # Models from your literature review
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name == 'Isolation Forest':
                # Unsupervised model
                model.fit(X_train)
                y_pred = model.predict(X_test)
                y_pred = (y_pred == -1).astype(int)  # Convert to binary
            else:
                # Supervised models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred
            }
            
            self.models[name] = model
            
            print(f"{name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return results
    
    def create_hybrid_model(self, X_train, X_test, y_train, y_test):
        """Create the proposed hybrid ensemble model"""
        print("\n=== Training Proposed Hybrid Model ===")
        
        # Supervised component
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Unsupervised component
        iso_model = IsolationForest(contamination=0.1, random_state=42)
        iso_model.fit(X_train)
        iso_scores = iso_model.decision_function(X_test)
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        
        # Deep learning component
        deep_model = self.create_deep_model(X_train.shape[1])
        deep_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        deep_pred_proba = deep_model.predict(X_test).flatten()
        
        # Ensemble combination (weighted average)
        w1, w2, w3 = 0.4, 0.3, 0.3  # Weights for RF, Isolation Forest, Deep Learning
        
        ensemble_scores = (w1 * rf_pred_proba + 
                          w2 * (1 - iso_scores_norm) +  # Invert isolation forest scores
                          w3 * deep_pred_proba)
        
        # Convert to binary predictions
        threshold = 0.5
        ensemble_pred = (ensemble_scores > threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        precision = precision_score(y_test, ensemble_pred, zero_division=0)
        recall = recall_score(y_test, ensemble_pred, zero_division=0)
        f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        
        hybrid_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': ensemble_pred,
            'scores': ensemble_scores
        }
        
        print(f"Proposed Model - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return hybrid_results
    
    def create_deep_model(self, input_dim):
        """Create deep learning model component"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model

# Visualization and Analysis
class ResultsAnalyzer:
    """
    Class for analyzing and visualizing results
    """
    
    def __init__(self):
        self.results = {}
        
    def create_comparison_table(self, results, hybrid_results):
        """Create comprehensive comparison table"""
        print("\n=== Model Performance Comparison ===")
        
        # Combine all results
        all_results = results.copy()
        all_results['Proposed Model'] = hybrid_results
        
        # Create DataFrame
        comparison_df = pd.DataFrame({
            'Model': list(all_results.keys()),
            'Accuracy': [all_results[model]['accuracy'] for model in all_results.keys()],
            'Precision': [all_results[model]['precision'] for model in all_results.keys()],
            'Recall': [all_results[model]['recall'] for model in all_results.keys()],
            'F1-Score': [all_results[model]['f1'] for model in all_results.keys()]
        })
        
        # Sort by F1-score
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        return comparison_df
    
    def plot_performance_comparison(self, comparison_df):
        """Create performance comparison visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(x=comparison_df['Model'], y=comparison_df[metric], 
                       name=metric, marker_color=colors[i], showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Model Performance Comparison Across Metrics",
            showlegend=False,
            height=600
        )
        
        fig.show()
        
    def plot_confusion_matrices(self, y_test, results, hybrid_results):
        """Plot confusion matrices for all models"""
        all_results = results.copy()
        all_results['Proposed Model'] = hybrid_results
        
        n_models = len(all_results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(all_results.items()):
            if i < len(axes):
                cm = confusion_matrix(y_test, result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{model_name}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        # Remove unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function"""
    print("=== CERT r6.2 Behavioral Analytics Pipeline ===")
    
    # Update this path to your CERT r6.2 dataset location
    DATA_PATH = "/path/to/your/cert_r6.2_dataset"  # UPDATE THIS PATH
    
    # Step 1: Data Processing
    processor = CERTDataProcessor(DATA_PATH)
    processor.load_data()
    processor.explore_data()
    processor.preprocess_data()
    
    # Step 2: Feature Extraction
    features_df = processor.extract_behavioral_features()
    labeled_data = processor.identify_insider_threats()
    
    # Step 3: Model Training
    models = BehavioralAnalyticsModels()
    X_train, X_test, y_train, y_test, feature_names = models.prepare_data(labeled_data)
    
    # Train baseline models
    baseline_results = models.train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Train proposed hybrid model
    hybrid_results = models.create_hybrid_model(X_train, X_test, y_train, y_test)
    
    # Step 4: Results Analysis
    analyzer = ResultsAnalyzer()
    comparison_table = analyzer.create_comparison_table(baseline_results, hybrid_results)
    
    # Visualizations
    analyzer.plot_performance_comparison(comparison_table)
    analyzer.plot_confusion_matrices(y_test, baseline_results, hybrid_results)
    
    print("\n=== Pipeline Execution Complete ===")
    print("Next steps:")
    print("1. Update DATA_PATH to your actual CERT r6.2 location")
    print("2. Fine-tune the hybrid model weights")
    print("3. Implement additional datasets (TWOS, your survey data)")
    print("4. Add more sophisticated feature engineering")
    
    return comparison_table, baseline_results, hybrid_results

if __name__ == "__main__":
    # Execute the complete pipeline
    results = main()