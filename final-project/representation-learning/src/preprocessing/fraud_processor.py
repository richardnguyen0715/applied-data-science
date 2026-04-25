"""Credit Card Fraud Detection data preprocessing"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class FraudProcessor:
    """Credit card fraud detection data processor"""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = self._init_scaler()
    
    def _init_scaler(self):
        """Initialize appropriate scaler"""
        scaling_method = self.config['preprocessing']['scaling'].lower()
        if scaling_method == 'standardscaler':
            return StandardScaler()
        elif scaling_method == 'robustscaler':
            return RobustScaler()
        else:
            return StandardScaler()
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load CSV data"""
        df = pd.read_csv(csv_path)
        return df
    
    def preprocess(self, df: pd.DataFrame, fit_scaler: bool = True) -> tuple:
        """Preprocess data"""
        df = df.copy()
        
        # Drop specified features
        drop_features = self.config['preprocessing']['drop_features']
        df = df.drop(columns=drop_features, errors='ignore')
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X) if fit_scaler else self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X_scaled, y
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        test_size = self.config['preprocessing']['test_size']
        val_size = self.config['preprocessing']['val_size']
        random_state = self.config['preprocessing']['random_state']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        train_size = 1 - val_size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, train_size=train_size, random_state=random_state, stratify=y_temp
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
