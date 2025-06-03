import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class HeartDiseasePredictor:
    def __init__(self, dataset_path):
        """
        Inisialisasi model prediksi penyakit jantung
        Args:
            dataset_path (str): Path ke file dataset heart.csv
        """
        self.dataset_path = dataset_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.features = None
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """
        Memuat dan mempersiapkan data untuk training
        """
        try:
            self.data = pd.read_csv(self.dataset_path)
            self.features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                             'thalach']
            
            X = self.data[self.features]
            y = self.data['target']
            
            # Normalisasi fitur
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(self.X_train, self.y_train)
            
        except Exception as e:
            raise Exception(f"Error saat mempersiapkan data: {str(e)}")

    def predict_heart_disease(self, input_features):
        """
        Memprediksi kemungkinan penyakit jantung berdasarkan fitur input
        Args:
            input_features (dict): Dictionary berisi nilai fitur
        Returns:
            dict: Hasil prediksi dan probabilitas terhadap penyakit jantung
        """
        try:
            if not all(feature in input_features for feature in self.features):
                raise ValueError("Fitur input tidak lengkap")
            
            input_df = pd.DataFrame([input_features])
            input_scaled = self.scaler.transform(input_df)
            
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0]
            
            return {
                'prediction': int(prediction),
                'probability': round(float(probability[1]), 3)  # selalu tampilkan probabilitas terkena penyakit (kelas 1)
            }
            
        except Exception as e:
            raise Exception(f"Error saat melakukan prediksi: {str(e)}")

    def get_model_metrics(self):
        """
        Mendapatkan metrik performa model
        Returns:
            dict: Metrik performa model
        """
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
        """
        Mendapatkan tingkat kepentingan fitur
        Returns:
            dict: Daftar kepentingan fitur
        """
        try:
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.features, importance))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            raise Exception(f"Error saat menghitung kepentingan fitur: {str(e)}")
