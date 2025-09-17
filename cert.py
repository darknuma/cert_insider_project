"""
CERT Insider Threat Detection Pipeline (v6.2)
- Uses DuckDB for efficient feature extraction
- Baseline models + Proposed Hybrid Model
- Results comparison and visualization
"""

import duckdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


# =============================
# Data Processing
# =============================

class CERTDataProcessor:
    """
    Data processor for CERT Insider Threat r6.2 dataset
    with DuckDB-powered feature extraction
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.con = duckdb.connect(database=':memory:')  # in-memory DB
        self.features_df = None
        self.insiders_df = None

    def load_data(self):
        """Register CSV logs in DuckDB"""
        print("\n=== Loading CERT r6.2 data into DuckDB ===")

        # Force all columns to VARCHAR to avoid auto-detection issues
        self.con.execute(f"""
            CREATE TABLE logon AS 
            SELECT * FROM read_csv_auto('{self.data_path}/logon.csv', ALL_VARCHAR=TRUE);
        """)
        self.con.execute(f"""
            CREATE TABLE email AS 
            SELECT * FROM read_csv_auto('{self.data_path}/email.csv', ALL_VARCHAR=TRUE);
        """)
        self.con.execute(f"""
            CREATE TABLE http AS 
            SELECT * FROM read_csv_auto('{self.data_path}/http.csv', ALL_VARCHAR=TRUE);
        """)
        self.con.execute(f"""
            CREATE TABLE file_access AS 
            SELECT * FROM read_csv_auto('{self.data_path}/file.csv', ALL_VARCHAR=TRUE);
        """)

        print("Tables registered:", self.con.execute("SHOW TABLES").fetchall())

    def extract_behavioral_features(self):
        """Aggregate user behaviours into feature table with proper timestamp parsing"""
        print("\n=== Extracting Features with DuckDB ===")

        # Logon activity
        logon_features = self.con.execute("""
            SELECT 
                user,
                DATE(STRPTIME(date, '%m/%d/%Y %H:%M:%S')) AS day,
                COUNT(*) AS login_count
            FROM logon
            GROUP BY user, day
        """).fetchdf()

        # Email activity
        email_features = self.con.execute("""
            SELECT 
                user,
                DATE(STRPTIME(date, '%m/%d/%Y %H:%M:%S')) AS day,
                COUNT(*) AS email_count
            FROM email
            GROUP BY user, day
        """).fetchdf()

        # File access activity
        file_features = self.con.execute("""
            SELECT 
                user,
                DATE(STRPTIME(date, '%m/%d/%Y %H:%M:%S')) AS day,
                COUNT(*) AS file_access_count
            FROM file_access
            GROUP BY user, day
        """).fetchdf()

        # HTTP activity
        http_features = self.con.execute("""
            SELECT 
                user,
                DATE(STRPTIME(date, '%m/%d/%Y %H:%M:%S')) AS day,
                COUNT(*) AS http_count
            FROM http
            GROUP BY user, day
        """).fetchdf()

        # Merge all features
        features_df = (logon_features
            .merge(email_features, on=["user", "day"], how="left")
            .merge(file_features, on=["user", "day"], how="left")
            .merge(http_features, on=["user", "day"], how="left")
        ).fillna(0)

        # Add placeholder labels
        features_df["is_insider_threat"] = 0
        features_df["anomaly_score"] = 0.0

        self.features_df = features_df
        print("Features shape:", features_df.shape)
        return features_df

    def identify_insider_threats(self):
        """Tag insider users from answers/insider.csv"""
        print("\n=== Tagging Insider Threats ===")

        insiders_path = f"{self.data_path}answers/insider.csv"
        self.insiders_df = pd.read_csv(insiders_path)
        print("Loaded insider list:", self.insiders_df.shape)

        # Expect a column named 'user' in insider.csv
        insider_users = set(self.insiders_df['user'].astype(str).unique())

        # Apply labels
        self.features_df['is_insider_threat'] = (
            self.features_df['user'].astype(str).isin(insider_users).astype(int)
        )

        print("Total insiders labeled:", self.features_df['is_insider_threat'].sum())
        return self.features_df


# =============================
# Behavioral Analytics Models
# =============================

class BehavioralAnalyticsModels:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    def prepare_data(self, features_df):
        """Prepare data for ML"""
        X = features_df.drop(['user', 'is_insider_threat', 'anomaly_score'], axis=1)
        y = features_df['is_insider_threat']
        X = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        print(f"Insiders in train: {y_train.sum()}, test: {y_test.sum()}")
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

    def train_baseline_models(self, X_train, X_test, y_train, y_test):
        """Train baseline models"""
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
                model.fit(X_train)
                y_pred = (model.predict(X_test) == -1).astype(int)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'predictions': y_pred
            }
            self.models[name] = model
        return results

    def create_hybrid_model(self, X_train, X_test, y_train, y_test):
        """Hybrid model: RF + Isolation Forest + Deep NN"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred_proba = rf.predict_proba(X_test)[:, 1]

        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(X_train)
        iso_scores = iso.decision_function(X_test)
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

        deep = self.create_deep_model(X_train.shape[1])
        deep.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        deep_pred_proba = deep.predict(X_test).flatten()

        ensemble_scores = 0.4*rf_pred_proba + 0.3*(1 - iso_scores_norm) + 0.3*deep_pred_proba
        ensemble_pred = (ensemble_scores > 0.5).astype(int)

        return {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, zero_division=0),
            'recall': recall_score(y_test, ensemble_pred, zero_division=0),
            'f1': f1_score(y_test, ensemble_pred, zero_division=0),
            'predictions': ensemble_pred,
            'scores': ensemble_scores
        }

    def create_deep_model(self, input_dim):
        """Simple Deep NN"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model


# =============================
# Results Analysis
# =============================

class ResultsAnalyzer:
    def create_comparison_table(self, results, hybrid_results):
        all_results = results.copy()
        all_results['Proposed Model'] = hybrid_results
        return pd.DataFrame({
            'Model': list(all_results.keys()),
            'Accuracy': [all_results[m]['accuracy'] for m in all_results],
            'Precision': [all_results[m]['precision'] for m in all_results],
            'Recall': [all_results[m]['recall'] for m in all_results],
            'F1-Score': [all_results[m]['f1'] for m in all_results]
        }).sort_values('F1-Score', ascending=False)

    def plot_confusion_matrices(self, y_test, results, hybrid_results):
        all_results = results.copy()
        all_results['Proposed Model'] = hybrid_results
        n_models = len(all_results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, (name, res) in enumerate(all_results.items()):
            if i < len(axes):
                cm = confusion_matrix(y_test, res['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(name)
        plt.tight_layout()
        plt.show()


# =============================
# Main Execution
# =============================

def main():
    print("=== CERT r6.2 Insider Threat Pipeline ===")
    DATA_PATH = "r6.2/"  # << update path if needed

    processor = CERTDataProcessor(DATA_PATH)
    processor.load_data()
    features = processor.extract_behavioral_features()
    labeled = processor.identify_insider_threats()

    models = BehavioralAnalyticsModels()
    X_train, X_test, y_train, y_test, feature_names = models.prepare_data(labeled)

    baseline = models.train_baseline_models(X_train, X_test, y_train, y_test)
    hybrid = models.create_hybrid_model(X_train, X_test, y_train, y_test)

    analyzer = ResultsAnalyzer()
    table = analyzer.create_comparison_table(baseline, hybrid)
    print(table.to_string(index=False, float_format="%.3f"))
    analyzer.plot_confusion_matrices(y_test, baseline, hybrid)

    print("\n=== Pipeline Complete ===")
    return table, baseline, hybrid


if __name__ == "__main__":
    main()
"""
CERT Insider Threat Detection Pipeline (v6.2)
- Uses DuckDB for efficient feature extraction
- Baseline models + Proposed Hybrid Model
- Results comparison and visualization
"""

import duckdb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


# =============================
# Data Processing
# =============================

class CERTDataProcessor:
    """
    Data processor for CERT Insider Threat r6.2 dataset
    with DuckDB-powered feature extraction
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.con = duckdb.connect(database=':memory:')  # in-memory DB
        self.features_df = None
        self.insiders_df = None

    def load_data(self):
        """Register CSV logs in DuckDB"""
        print("\n=== Loading CERT r6.2 data into DuckDB ===")

        # Force all columns to VARCHAR to avoid auto-detection issues
        self.con.execute(f"""
            CREATE TABLE logon AS 
            SELECT * FROM read_csv_auto('{self.data_path}/logon.csv', ALL_VARCHAR=TRUE);
        """)
        self.con.execute(f"""
            CREATE TABLE email AS 
            SELECT * FROM read_csv_auto('{self.data_path}/email.csv', ALL_VARCHAR=TRUE);
        """)
        self.con.execute(f"""
            CREATE TABLE http AS 
            SELECT * FROM read_csv_auto('{self.data_path}/http.csv', ALL_VARCHAR=TRUE);
        """)
        self.con.execute(f"""
            CREATE TABLE file_access AS 
            SELECT * FROM read_csv_auto('{self.data_path}/file.csv', ALL_VARCHAR=TRUE);
        """)

        print("Tables registered:", self.con.execute("SHOW TABLES").fetchall())

    def extract_behavioral_features(self):
        """Aggregate user behaviours into feature table with proper timestamp parsing"""
        print("\n=== Extracting Features with DuckDB ===")

        # Logon activity
        logon_features = self.con.execute("""
            SELECT 
                user,
                DATE(STRPTIME(date, '%m/%d/%Y %H:%M:%S')) AS day,
                COUNT(*) AS login_count
            FROM logon
            GROUP BY user, day
        """).fetchdf()

        # Email activity
        email_features = self.con.execute("""
            SELECT 
                user,
                DATE(STRPTIME(date, '%m/%d/%Y %H:%M:%S')) AS day,
                COUNT(*) AS email_count
            FROM email
            GROUP BY user, day
        """).fetchdf()

        # File access activity
        file_features = self.con.execute("""
            SELECT 
                user,
                DATE(STRPTIME(date, '%m/%d/%Y %H:%M:%S')) AS day,
                COUNT(*) AS file_access_count
            FROM file_access
            GROUP BY user, day
        """).fetchdf()

        # HTTP activity
        http_features = self.con.execute("""
            SELECT 
                user,
                DATE(STRPTIME(date, '%m/%d/%Y %H:%M:%S')) AS day,
                COUNT(*) AS http_count
            FROM http
            GROUP BY user, day
        """).fetchdf()

        # Merge all features
        features_df = (logon_features
            .merge(email_features, on=["user", "day"], how="left")
            .merge(file_features, on=["user", "day"], how="left")
            .merge(http_features, on=["user", "day"], how="left")
        ).fillna(0)

        # Add placeholder labels
        features_df["is_insider_threat"] = 0
        features_df["anomaly_score"] = 0.0

        self.features_df = features_df
        print("Features shape:", features_df.shape)
        return features_df

    def identify_insider_threats(self):
        """Tag insider users from answers/insider.csv"""
        print("\n=== Tagging Insider Threats ===")

        insiders_path = f"{self.data_path}/../answers/insider.csv" "r6.2/answers/insiders.csv"
        self.insiders_df = pd.read_csv(insiders_path)
        print("Loaded insider list:", self.insiders_df.shape)

        # Expect a column named 'user' in insider.csv
        insider_users = set(self.insiders_df['user'].astype(str).unique())

        # Apply labels
        self.features_df['is_insider_threat'] = (
            self.features_df['user'].astype(str).isin(insider_users).astype(int)
        )

        print("Total insiders labeled:", self.features_df['is_insider_threat'].sum())
        return self.features_df


# =============================
# Behavioral Analytics Models
# =============================

class BehavioralAnalyticsModels:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    def prepare_data(self, features_df):
        """Prepare data for ML"""
        X = features_df.drop(['user', 'is_insider_threat', 'anomaly_score'], axis=1)
        y = features_df['is_insider_threat']
        X = X.select_dtypes(include=[np.number])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        print(f"Insiders in train: {y_train.sum()}, test: {y_test.sum()}")
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

    def train_baseline_models(self, X_train, X_test, y_train, y_test):
        """Train baseline models"""
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
                model.fit(X_train)
                y_pred = (model.predict(X_test) == -1).astype(int)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'predictions': y_pred
            }
            self.models[name] = model
        return results

    def create_hybrid_model(self, X_train, X_test, y_train, y_test):
        """Hybrid model: RF + Isolation Forest + Deep NN"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred_proba = rf.predict_proba(X_test)[:, 1]

        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(X_train)
        iso_scores = iso.decision_function(X_test)
        iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

        deep = self.create_deep_model(X_train.shape[1])
        deep.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        deep_pred_proba = deep.predict(X_test).flatten()

        ensemble_scores = 0.4*rf_pred_proba + 0.3*(1 - iso_scores_norm) + 0.3*deep_pred_proba
        ensemble_pred = (ensemble_scores > 0.5).astype(int)

        return {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, zero_division=0),
            'recall': recall_score(y_test, ensemble_pred, zero_division=0),
            'f1': f1_score(y_test, ensemble_pred, zero_division=0),
            'predictions': ensemble_pred,
            'scores': ensemble_scores
        }

    def create_deep_model(self, input_dim):
        """Simple Deep NN"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model


# =============================
# Results Analysis
# =============================

class ResultsAnalyzer:
    def create_comparison_table(self, results, hybrid_results):
        all_results = results.copy()
        all_results['Proposed Model'] = hybrid_results
        return pd.DataFrame({
            'Model': list(all_results.keys()),
            'Accuracy': [all_results[m]['accuracy'] for m in all_results],
            'Precision': [all_results[m]['precision'] for m in all_results],
            'Recall': [all_results[m]['recall'] for m in all_results],
            'F1-Score': [all_results[m]['f1'] for m in all_results]
        }).sort_values('F1-Score', ascending=False)

    def plot_confusion_matrices(self, y_test, results, hybrid_results):
        all_results = results.copy()
        all_results['Proposed Model'] = hybrid_results
        n_models = len(all_results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, (name, res) in enumerate(all_results.items()):
            if i < len(axes):
                cm = confusion_matrix(y_test, res['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(name)
        plt.tight_layout()
        plt.show()


# =============================
# Main Execution
# =============================

def main():
    print("=== CERT r6.2 Insider Threat Pipeline ===")
    DATA_PATH = "r6.2/"  # << update path if needed

    processor = CERTDataProcessor(DATA_PATH)
    processor.load_data()
    features = processor.extract_behavioral_features()
    labeled = processor.identify_insider_threats()

    models = BehavioralAnalyticsModels()
    X_train, X_test, y_train, y_test, feature_names = models.prepare_data(labeled)

    baseline = models.train_baseline_models(X_train, X_test, y_train, y_test)
    hybrid = models.create_hybrid_model(X_train, X_test, y_train, y_test)

    analyzer = ResultsAnalyzer()
    table = analyzer.create_comparison_table(baseline, hybrid)
    print(table.to_string(index=False, float_format="%.3f"))
    analyzer.plot_confusion_matrices(y_test, baseline, hybrid)

    print("\n=== Pipeline Complete ===")
    return table, baseline, hybrid


if __name__ == "__main__":
    main()