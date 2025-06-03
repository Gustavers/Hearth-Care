import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class HeartDiseasePredictor:
    def __init__(self, dataset_path, model_path='saved_model.pkl', scaler_path='scaler.pkl'):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach']
        self.model = None
        self.scaler = None

        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self._load_model()
        else:
            self._load_and_prepare_data()
            self._save_model()

    def _load_and_prepare_data(self):
        try:
            self.data = pd.read_csv(self.dataset_path)

            X = self.data[self.features]
            y = self.data['target']

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)
        except Exception as e:
            raise Exception(f"Error saat mempersiapkan data: {str(e)}")

    def _save_model(self):
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
        except Exception as e:
            raise Exception(f"Error saat menyimpan model: {str(e)}")

    def _load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.data = pd.read_csv(self.dataset_path)
            X = self.data[self.features]
            y = self.data['target']
            X_scaled = self.scaler.transform(X)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        except Exception as e:
            raise Exception(f"Error saat memuat model: {str(e)}")

    def predict_heart_disease(self, input_features):
        try:
            if not all(feature in input_features for feature in self.features):
                raise ValueError("Fitur input tidak lengkap")

            input_df = pd.DataFrame([input_features])
            input_scaled = self.scaler.transform(input_df)

            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0]

            return {
                'prediction': int(prediction),
                'probability': round(float(probability[1]), 3)
            }

        except Exception as e:
            raise Exception(f"Error saat melakukan prediksi: {str(e)}")

    def get_model_metrics(self):
        try:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(self.y_test, y_pred)

            return {
                'accuracy': round(accuracy, 3),
                'precision': round(report['weighted avg']['precision'], 3),
                'recall': round(report['weighted avg']['recall'], 3),
                'f1_score': round(report['weighted avg']['f1-score'], 3),
                'confusion_matrix': conf_matrix.tolist()
            }
        except Exception as e:
            raise Exception(f"Error saat menghitung metrik: {str(e)}")

    def get_feature_importance(self):
        try:
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.features, importance))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            raise Exception(f"Error saat menghitung kepentingan fitur: {str(e)}")
