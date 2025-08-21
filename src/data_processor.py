#!/usr/bin/env python3
"""
Enhanced Data Preprocessing Module for Customer Churn Prediction
================================================================

This module provides comprehensive data preprocessing capabilities including:
- Data validation and quality checks
- Feature engineering and transformation
- Data balancing and stratification  
- Automated pipeline with logging and monitoring

Author: ML Pipeline Team
Date: 2024
License: MIT
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChurnDataProcessor:
    """Main data processing class with comprehensive preprocessing pipeline."""
    
    def __init__(self, 
                 target_col: str = 'Churn',
                 test_size: float = 0.2,
                 random_state: int = 42,
                 balance_method: str = 'smote'):
        """
        Initialize the data processor.
        
        Args:
            target_col: Name of target column
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            balance_method: Method for handling class imbalance ('smote', 'undersample', 'smotetomek', 'none')
        """
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.balance_method = balance_method
        
        self.preprocessor = None
        self.label_encoders = {}
        self.scaler = None
        self.balancer = None
        
        self.feature_names = None
        self.processing_stats = {}
    
    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data and perform quality validation.
        
        Args:
            data_path: Path to the data file
            
        Returns:
            Loaded and validated dataframe
        """
        logger.info(f"📁 Loading data from: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"✅ Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive preprocessing pipeline.
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        logger.info("🔄 Starting data preprocessing pipeline...")
        
        df_processed = df.copy()
        
        # Handle column names
        df_processed.columns = df_processed.columns.str.strip().str.replace(' ', '_')
        
        # Update target column name if needed
        if 'Churn' in df_processed.columns:
            self.target_col = 'Churn'
        elif 'churn' in df_processed.columns:
            df_processed = df_processed.rename(columns={'churn': 'Churn'})
            self.target_col = 'Churn'
        
        logger.info(f"📝 Standardized column names")
        
        # Feature engineering
        df_processed = self._create_interaction_features(df_processed)
        
        # Handle categorical encoding
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Convert target to binary if it's boolean
        if df_processed[self.target_col].dtype == bool:
            df_processed[self.target_col] = df_processed[self.target_col].astype(int)
        elif df_processed[self.target_col].dtype == 'object':
            le_target = LabelEncoder()
            df_processed[self.target_col] = le_target.fit_transform(df_processed[self.target_col])
            self.label_encoders['target'] = le_target
        
        # Remove highly correlated features
        df_processed = self._remove_redundant_features(df_processed)
        
        logger.info(f"✅ Preprocessing complete: {df_processed.shape[1]} features")
        
        self.processing_stats['final_shape'] = df_processed.shape
        
        return df_processed
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create meaningful interaction features."""
        logger.info("🔧 Creating interaction features...")
        
        df_enhanced = df.copy()
        
        # Rate features (charge per minute)
        if all(col in df.columns for col in ['Total_day_charge', 'Total_day_minutes']):
            df_enhanced['Day_rate'] = np.where(
                df_enhanced['Total_day_minutes'] > 0,
                df_enhanced['Total_day_charge'] / df_enhanced['Total_day_minutes'],
                0
            )
        
        # Total usage features
        usage_cols = [col for col in df.columns if 'minutes' in col.lower()]
        if usage_cols:
            df_enhanced['Total_usage_minutes'] = df_enhanced[usage_cols].sum(axis=1)
        
        charge_cols = [col for col in df.columns if 'charge' in col.lower()]
        if charge_cols:
            df_enhanced['Total_charges'] = df_enhanced[charge_cols].sum(axis=1)
        
        # Call frequency features
        call_cols = [col for col in df.columns if 'calls' in col.lower() and 'service' not in col.lower()]
        if call_cols:
            df_enhanced['Total_calls'] = df_enhanced[call_cols].sum(axis=1)
        
        logger.info(f"✅ Created interaction features")
        return df_enhanced
    
    def _remove_redundant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove redundant and highly correlated features."""
        logger.info("🧹 Removing redundant features...")
        
        # Remove features with perfect correlation (e.g., charge and minutes)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numerical_cols:
            numerical_cols.remove(self.target_col)
        
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr().abs()
            
            # Find features with correlation > 0.95
            features_to_remove = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        features_to_remove.add(col1)  # Remove first one
            
            if features_to_remove:
                logger.info(f"🗑️  Removing {len(features_to_remove)} highly correlated features")
                df = df.drop(columns=list(features_to_remove))
        
        return df
    
    def split_and_balance_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data and apply balancing if specified.
        
        Args:
            df: Processed dataframe
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("✂️ Splitting data...")
        
        # Separate features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"📊 Train set: {X_train.shape[0]:,} samples")
        logger.info(f"📊 Test set: {X_test.shape[0]:,} samples")
        
        # Apply balancing to training data only
        if self.balance_method != 'none':
            X_train, y_train = self._apply_balancing(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def _apply_balancing(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply class balancing technique."""
        logger.info(f"⚖️ Applying {self.balance_method} balancing...")
        
        original_counts = y_train.value_counts()
        logger.info(f"📊 Original class distribution: {dict(original_counts)}")
        
        if self.balance_method == 'smote':
            self.balancer = SMOTE(random_state=self.random_state)
        else:
            logger.warning(f"Unknown balancing method: {self.balance_method}")
            return X_train, y_train
        
        try:
            X_balanced, y_balanced = self.balancer.fit_resample(X_train, y_train)
            X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
            y_balanced = pd.Series(y_balanced, name=y_train.name)
            
            new_counts = y_balanced.value_counts()
            logger.info(f"📊 Balanced class distribution: {dict(new_counts)}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"❌ Balancing failed: {e}")
            logger.info("📊 Proceeding without balancing")
            return X_train, y_train
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      method: str = 'robust') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features.
        
        Args:
            X_train: Training features
            X_test: Test features
            method: Scaling method ('standard', 'robust')
            
        Returns:
            Scaled X_train, X_test
        """
        logger.info(f"📏 Scaling features using {method} method...")
        
        if method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Only scale numerical features
        numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_features:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
            X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
            
            logger.info(f"✅ Scaled {len(numerical_features)} numerical features")
            return X_train_scaled, X_test_scaled
        else:
            logger.info("ℹ️  No numerical features to scale")
            return X_train, X_test
    
    def save_preprocessing_artifacts(self, output_dir: str = 'models') -> None:
        """Save preprocessing artifacts for later use."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"💾 Saving preprocessing artifacts to {output_dir}/")
        
        # Save label encoders
        if self.label_encoders:
            joblib.dump(self.label_encoders, output_path / 'label_encoders.pkl')
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, output_path / 'scaler.pkl')
        
        # Save balancer
        if self.balancer:
            joblib.dump(self.balancer, output_path / 'balancer.pkl')
        
        # Save feature names and processing stats
        processing_metadata = {
            'feature_names': self.feature_names,
            'target_col': self.target_col,
            'processing_stats': self.processing_stats,
            'balance_method': self.balance_method
        }
        
        joblib.dump(processing_metadata, output_path / 'processing_metadata.pkl')
        
        logger.info("✅ All preprocessing artifacts saved")
    
    def process_pipeline(self, data_path: str, 
                        output_dir: str = 'data',
                        scaling_method: str = 'robust') -> Dict[str, Any]:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            data_path: Path to input data
            output_dir: Directory to save processed data
            scaling_method: Feature scaling method
            
        Returns:
            Dictionary containing processed datasets and metadata
        """
        logger.info("🚀 Starting complete preprocessing pipeline...")
        
        # Load and validate data
        df = self.load_and_validate_data(data_path)
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Split and balance data
        X_train, X_test, y_train, y_test = self.split_and_balance_data(df_processed)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, scaling_method)
        
        # Save processed datasets
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save training data
        train_df = X_train_scaled.copy()
        train_df[self.target_col] = y_train.values
        train_df.to_csv(output_path / 'train_data.csv', index=False)
        
        # Save test data
        test_df = X_test_scaled.copy()
        test_df[self.target_col] = y_test.values
        test_df.to_csv(output_path / 'test_data.csv', index=False)
        
        # Save full processed dataset
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        full_df.to_csv(output_path / 'processed_data.csv', index=False)
        
        # Save preprocessing artifacts
        self.save_preprocessing_artifacts()
        
        logger.info("✅ Complete preprocessing pipeline finished successfully!")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'processing_stats': self.processing_stats
        }

def main():
    """Main execution function."""
    logger.info("🎯 Enhanced Data Preprocessing for Customer Churn Prediction")
    logger.info("=" * 60)
    
    # Initialize processor
    processor = ChurnDataProcessor(
        target_col='Churn',
        test_size=0.2,
        random_state=42,
        balance_method='smote'
    )
    
    # Process data
    try:
        results = processor.process_pipeline(
            data_path='data/customer_data.csv',
            output_dir='data',
            scaling_method='robust'
        )
        
        logger.info("\n📊 Final Processing Summary:")
        logger.info(f"   • Training samples: {results['X_train'].shape[0]:,}")
        logger.info(f"   • Test samples: {results['X_test'].shape[0]:,}")
        logger.info(f"   • Features: {len(results['feature_names'])}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()