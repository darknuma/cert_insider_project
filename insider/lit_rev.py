"""
Implementation of Specific Methods from Literature Review
Based on the papers cited in your behavioral analytics research
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.ensemble import BalancedRandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

class HanEtAlMethod:
    """
    Implementation of Han et al. method using machine learning with preprocessing
    Reference: "A Study on Detection of Malicious Behavior Based on Host Process Data Using Machine Learning"
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.pca_reducers = {}
        self.results = {}
        
    def apply_preprocessing(self, X_train, X_test, y_train, method='smote'):
        """Apply preprocessing techniques like SMOTE, PCA, etc."""
        print(f"Applying {method.upper()} preprocessing...")
        
        # Apply SMOTE or ADASYN for balancing
        if method.lower() == 'smote':
            sampler = SMOTE(random_state=42)
        elif method.lower() == 'adasyn':
            sampler = ADASYN(random_state=42)
        else:
            sampler = None
            
        if sampler:
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
            print(f"Original training samples: {len(y_train)}")
            print(f"After {method}: {len(y_train_balanced)}")
            print(f"Class distribution: {np.bincount(y_train_balanced)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
            
        return X_train_balanced, y_train_balanced
    
    def apply_pca_reduction(self, X_train, X_test, n_components=0.95):
        """Apply PCA for dimensionality reduction"""
        print(f"Applying PCA with {n_components} variance retention...")
        
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        print(f"Original features: {X_train.shape[1]}")
        print(f"PCA components: {X_train_pca.shape[1]}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        return X_train_pca, X_test_pca, pca
    
    def train_han_models(self, X_train, X_test, y_train, y_test):
        """Train models as described in Han et al. paper"""
        print("\n=== Han et al. Method Implementation ===")
        
        # Models mentioned in the paper
        models = {
            'KNN_Han': KNeighborsClassifier(n_neighbors=5),
            'NaiveBayes_Han': GaussianNB(),
            'RandomForest_Han': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        # Test with different preprocessing approaches
        preprocessing_methods = ['original', 'smote', 'adasyn']
        
        for prep_method in preprocessing_methods:
            print(f"\n--- Testing with {prep_method} preprocessing ---")
            
            if prep_method == 'original':
                X_train_prep, y_train_prep = X_train, y_train
            else:
                X_train_prep, y_train_prep = self.apply_preprocessing(X_train, X_test, y_train, prep_method)
            
            # Apply PCA
            X_train_pca, X_test_pca, pca = self.apply_pca_reduction(X_train_prep, X_test)
            
            for model_name, model in models.items():
                full_name = f"{model_name}_{prep_method}"
                print(f"Training {full_name}...")
                
                # Train model
                model.fit(X_train_pca, y_train_prep)
                y_pred = model.predict(X_test_pca)
                y_pred_proba = model.predict_proba(X_test_pca)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
                
                results[full_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'predictions': y_pred
                }
                
                print(f"  Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        self.results.update(results)
        return results

class JanjuaEtAlMethod:
    """
    Implementation of Janjua et al. method for insider threat detection
    Reference: "Handling insider threat through supervised machine learning techniques"
    Focus on email classification approach
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def extract_email_features(self, email_data):
        """Extract email-specific features as described in the paper"""
        print("Extracting email-based features...")
        
        email_features = []
        
        for user in email_data['user'].unique():
            user_emails = email_data[email_data['user'] == user]
            
            features = {
                'user': user,
                'total_emails': len(user_emails),
                'unique_recipients': user_emails['to'].nunique() if 'to' in user_emails.columns else 0,
                'avg_emails_per_day': len(user_emails) / max(1, user_emails['date'].dt.date.nunique()) if 'date' in user_emails.columns else 0,
            }
            
            # Email content analysis (if available)
            if 'content' in user_emails.columns:
                # Simple content features
                user_emails['content_length'] = user_emails['content'].str.len().fillna(0)
                features['avg_email_length'] = user_emails['content_length'].mean()
                features['max_email_length'] = user_emails['content_length'].max()
                
                # Keyword-based features (simplified)
                keywords = ['confidential', 'secret', 'urgent', 'attachment', 'forward']
                for keyword in keywords:
                    features[f'{keyword}_mentions'] = user_emails['content'].str.contains(keyword, case=False, na=False).sum()
            
            # Time-based patterns
            if 'date' in user_emails.columns:
                user_emails['hour'] = user_emails['date'].dt.hour
                features['after_hours_emails'] = len(user_emails[(user_emails['hour'] < 8) | (user_emails['hour'] > 18)])
                features['weekend_emails'] = len(user_emails[user_emails['date'].dt.dayofweek.isin([5, 6])])
            
            # External communication patterns
            if 'to' in user_emails.columns:
                external_emails = user_emails[~user_emails['to'].str.contains('@dtaa.com', na=False)]
                features['external_email_ratio'] = len(external_emails) / max(1, len(user_emails))
            
            email_features.append(features)
        
        return pd.DataFrame(email_features).fillna(0)
    
    def train_janjua_models(self, X_train, X_test, y_train, y_test):
        """Train models as described in Janjua et al. paper"""
        print("\n=== Janjua et al. Method Implementation ===")
        
        # Models mentioned in the paper
        models = {
            'AdaBoost_Janjua': AdaBoostClassifier(n_estimators=100, random_state=42),
            'NaiveBayes_Janjua': GaussianNB(),
            'LogisticRegression_Janjua': SVC(kernel='linear', probability=True, random_state=42),  # SVM with linear kernel as proxy
            'KNN_Janjua': KNeighborsClassifier(n_neighbors=5),
            'SVM_Janjua': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        results = {}
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred
            }
            
            print(f"  Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        self.results.update(results)
        return results

class MehmoodEtAlMethod:
    """
    Implementation of Mehmood et al. method using ensemble learning
    Reference: "Privilege Escalation Attack Detection and Mitigation in Cloud using Machine Learning"
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_mehmood_models(self, X_train, X_test, y_train, y_test):
        """Train ensemble models as described in Mehmood et al. paper"""
        print("\n=== Mehmood et al. Method Implementation ===")
        
        # Models mentioned in the paper with their reported accuracies
        models = {
            'RandomForest_Mehmood': RandomForestClassifier(n_estimators=100, random_state=42),
            'AdaBoost_Mehmood': AdaBoostClassifier(n_estimators=100, random_state=42),
            'XGBoost_Mehmood': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM_Mehmood': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred
            }
            
            print(f"  Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        self.results.update(results)
        return results

class AlshehriEtAlMethod:
    """
    Implementation of Alshehri et al. RNN-LSTM method
    Reference: RNN-LSTM model for cyberattack detection with user behavior analytics
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_sequence_data(self, features_df, sequence_length=10):
        """Prepare data for LSTM input (sequences of user behavior)"""
        print(f"Preparing sequence data with length {sequence_length}...")
        
        sequences = []
        labels = []
        
        for user in features_df['user'].unique():
            user_data = features_df[features_df['user'] == user].drop(['user'], axis=1)
            
            if len(user_data) >= sequence_length:
                for i in range(len(user_data) - sequence_length + 1):
                    seq = user_data.iloc[i:i+sequence_length].drop(['is_insider_threat'], axis=1).values
                    label = user_data.iloc[i+sequence_length-1]['is_insider_threat']
                    sequences.append(seq)
                    labels.append(label)
        
        return np.array(sequences), np.array(labels)
    
    def build_lstm_model(self, input_shape):
        """Build RNN-LSTM model as described in the paper"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def train_alshehri_model(self, X_train, X_test, y_train, y_test):
        """Train the RNN-LSTM model"""
        print("\n=== Alshehri et al. RNN-LSTM Method Implementation ===")
        
        # Reshape data for LSTM if needed
        if len(X_train.shape) == 2:
            # Convert to sequences (simplified approach)
            X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        else:
            X_train_lstm, X_test_lstm = X_train, X_test
        
        # Build and train model
        model = self.build_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
        
        print("Training RNN-LSTM model...")
        history = model.fit(X_train_lstm, y_train,
                           epochs=50,
                           batch_size=32,
                           validation_split=0.2,
                           verbose=0)
        
        # Make predictions
        y_pred_proba = model.predict(X_test_lstm).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'RNN_LSTM_Alshehri': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred
            }
        }
        
        print(f"RNN-LSTM Results - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        
        self.model = model
        return results

class BinSarhanAltwaijryMethod:
    """
    Implementation of Bin Sarhan and Altwaijry method using feature synthesis
    Reference: "Insider Threat Detection Using Machine Learning Approach"
    """
    
    def __init__(self):
        self.models = {}
        self.feature_extractor = None
        
    def deep_feature_synthesis(self, features_df):
        """Simplified deep feature synthesis approach"""
        print("Applying deep feature synthesis...")
        
        # Create additional derived features
        synthesized_features = features_df.copy()
        
        # Ratio features
        numeric_cols = synthesized_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'is_insider_threat' and synthesized_features[col].sum() > 0:
                synthesized_features[f'{col}_ratio_to_total'] = synthesized_features[col] / (synthesized_features[col].sum() + 1e-8)
        
        # Interaction features (simplified)
        if 'total_logons' in synthesized_features.columns and 'total_file_events' in synthesized_features.columns:
            synthesized_features['logon_file_interaction'] = synthesized_features['total_logons'] * synthesized_features['total_file_events']
        
        if 'after_hours_logons' in synthesized_features.columns and 'total_logons' in synthesized_features.columns:
            synthesized_features['after_hours_ratio'] = synthesized_features['after_hours_logons'] / (synthesized_features['total_logons'] + 1)
        
        print(f"Feature synthesis completed: {synthesized_features.shape[1]} features")
        return synthesized_features
    
    def train_binsarhan_models(self, X_train, X_test, y_train, y_test):
        """Train models with feature synthesis and PCA"""
        print("\n=== Bin Sarhan & Altwaijry Method Implementation ===")
        
        # Apply PCA as mentioned in the paper
        pca = PCA(n_components=0.95, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        print(f"PCA reduction: {X_train.shape[1]} -> {X_train_pca.shape[1]} features")
        
        # Test both with and without SMOTE as mentioned in paper
        models_config = [
            ('SVM_BinSarhan_NoSMOTE', SVC(kernel='rbf', probability=True, random_state=42), False),
            ('SVM_BinSarhan_SMOTE', SVC(kernel='rbf', probability=True, random_state=42), True),
            ('RandomForest_BinSarhan', RandomForestClassifier(n_estimators=100, random_state=42), True)
        ]
        
        results = {}
        
        for model_name, model, use_smote in models_config:
            print(f"Training {model_name}...")
            
            if use_smote:
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)
            else:
                X_train_balanced, y_train_balanced = X_train_pca, y_train
            
            model.fit(X_train_balanced, y_train_balanced)
            y_pred = model.predict(X_test_pca)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred
            }
            
            print(f"  Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return results

class LiteratureReviewComparator:
    """
    Main class to coordinate all literature review method implementations
    """
    
    def __init__(self):
        self.all_results = {}
        self.comparison_table = None
        
    def run_all_methods(self, X_train, X_test, y_train, y_test, email_data=None):
        """Run all literature review methods and compile results"""
        print("=== Running All Literature Review Methods ===\n")
        
        # Han et al. method
        han_method = HanEtAlMethod()
        han_results = han_method.train_han_models(X_train, X_test, y_train, y_test)
        self.all_results.update(han_results)
        
        # Janjua et al. method
        janjua_method = JanjuaEtAlMethod()
        janjua_results = janjua_method.train_janjua_models(X_train, X_test, y_train, y_test)
        self.all_results.update(janjua_results)
        
        # Mehmood et al. method
        mehmood_method = MehmoodEtAlMethod()
        mehmood_results = mehmood_method.train_mehmood_models(X_train, X_test, y_train, y_test)
        self.all_results.update(mehmood_results)
        
        # Alshehri et al. RNN-LSTM method
        alshehri_method = AlshehriEtAlMethod()
        alshehri_results = alshehri_method.train_alshehri_model(X_train, X_test, y_train, y_test)
        self.all_results.update(alshehri_results)
        
        # Bin Sarhan & Altwaijry method
        binsarhan_method = BinSarhanAltwaijryMethod()
        binsarhan_results = binsarhan_method.train_binsarhan_models(X_train, X_test, y_train, y_test)
        self.all_results.update(binsarhan_results)
        
        return self.all_results
    
    def create_comprehensive_comparison(self, proposed_model_results=None):
        """Create comprehensive comparison table with all methods"""
        print("\n=== Comprehensive Literature Review Comparison ===")
        
        all_methods = self.all_results.copy()
        if proposed_model_results:
            all_methods['Proposed_Model'] = proposed_model_results
        
        # Create comparison DataFrame
        comparison_data = []
        for method_name, results in all_methods.items():
            comparison_data.append({
                'Method': method_name,
                'Paper_Reference': self.get_paper_reference(method_name),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        self.comparison_table = comparison_df
        return comparison_df
    
    def get_paper_reference(self, method_name):
        """Get paper reference for each method"""
        references = {
            'Han': 'Han et al. (2023)',
            'Janjua': 'Janjua et al. (2020)',
            'Mehmood': 'Mehmood et al. (2023)',
            'Alshehri': 'Alshehri et al. (2021)',
            'BinSarhan': 'Bin Sarhan & Altwaijry (2022)',
            'Proposed': 'Your Proposed Method'
        }
        
        for key, ref in references.items():
            if key in method_name:
                return ref
        return 'Unknown Reference'

# Usage example function
def main_literature_implementation(features_df):
    """
    Main function to run all literature review implementations
    """
    print("=== Literature Review Methods Implementation ===")
    
    # Prepare data
    X = features_df.drop(['user', 'is_insider_threat'], axis=1, errors='ignore')
    y = features_df['is_insider_threat'] if 'is_insider_threat' in features_df.columns else None
    
    if y is None:
        print("Error: 'is_insider_threat' column not found in features_df")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Run all literature review methods
    comparator = LiteratureReviewComparator()
    all_results = comparator.run_all_methods(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Create comparison table
    comparison_table = comparator.create_comprehensive_comparison()
    
    return comparison_table, all_results, comparator

if __name__ == "__main__":
    print("Literature Review Methods Implementation Ready!")
    print("Use main_literature_implementation(your_features_df) to run all methods")