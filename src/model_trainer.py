#!/usr/bin/env python3
"""
Advanced Model Training Module for Customer Churn Prediction
===========================================================

This module provides comprehensive model training capabilities including:
- Multiple algorithm evaluation and comparison
- Cross-validation and performance metrics
- Model interpretability and feature importance analysis
- Automated model selection and ensemble methods

Author: ML Pipeline Team
Date: 2024
License: MIT
"""

import pandas as pd
import numpy as np
import logging
import joblib
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and metrics calculation."""
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Add probabilistic metrics if probabilities are available
        if y_pred_proba is not None:
            try:
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                    y_scores = y_pred_proba[:, 1]
                else:
                    y_scores = y_pred_proba.ravel()
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        return metrics
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series, 
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"🔄 Performing {cv_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
                cv_results[metric] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores.tolist()
                }
            except Exception as e:
                logger.warning(f"Could not calculate {metric}: {e}")
        
        return cv_results

class ChurnModelTrainer:
    """Main class for comprehensive model training and evaluation."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_results = {}
        
        self.evaluator = ModelEvaluator()
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all available models."""
        logger.info("🔧 Initializing model portfolio...")
        
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'GradientBoosting': GradientBoostingClassifier(random_state=self.random_state, n_estimators=100),
            'LogisticRegression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        logger.info(f"✅ Initialized {len(self.models)} models")
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train and evaluate a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Model results dictionary
        """
        if model_name not in self.models:
            logger.error(f"❌ Model {model_name} not available")
            return {}
        
        logger.info(f"🚀 Training {model_name}...")
        start_time = time.time()
        
        model = self.models[model_name]
        
        # Train the model
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Get prediction probabilities
        try:
            y_pred_proba_train = model.predict_proba(X_train)
            y_pred_proba_test = model.predict_proba(X_test)
        except:
            y_pred_proba_train = None
            y_pred_proba_test = None
        
        # Calculate metrics
        train_metrics = self.evaluator.calculate_metrics(y_train, y_pred_train, y_pred_proba_train)
        test_metrics = self.evaluator.calculate_metrics(y_test, y_pred_test, y_pred_proba_test)
        
        # Cross-validation
        cv_results = self.evaluator.cross_validate_model(model, X_train, y_train)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X_train.columns, abs(model.coef_[0])))
        
        # Store results
        results = {
            'model': model,
            'model_name': model_name,
            'training_time': training_time,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_results': cv_results,
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred_test)
        }
        
        self.trained_models[model_name] = model
        self.model_results[model_name] = results
        
        logger.info(f"✅ {model_name} training complete in {training_time:.2f}s")
        logger.info(f"📊 Test F1-Score: {test_metrics.get('f1_score', 0):.4f}")
        
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train and evaluate all available models.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of all model results
        """
        logger.info("🎯 Starting comprehensive model training...")
        
        # Train each model
        all_results = {}
        for model_name in self.models.keys():
            try:
                results = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                continue
        
        # Generate comparison report
        self._generate_model_comparison(all_results)
        
        return all_results
    
    def _generate_model_comparison(self, results: Dict[str, Any]) -> None:
        """Generate and display model comparison report."""
        logger.info("\n" + "="*60)
        logger.info("📊 MODEL COMPARISON REPORT")
        logger.info("="*60)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, result in results.items():
            if result:
                test_metrics = result.get('test_metrics', {})
                cv_results = result.get('cv_results', {})
                
                comparison_data.append({
                    'Model': model_name,
                    'Test_Accuracy': test_metrics.get('accuracy', 0),
                    'Test_F1': test_metrics.get('f1_score', 0),
                    'Test_Precision': test_metrics.get('precision', 0),
                    'Test_Recall': test_metrics.get('recall', 0),
                    'Test_ROC_AUC': test_metrics.get('roc_auc', 0),
                    'CV_F1_Mean': cv_results.get('f1_weighted', {}).get('mean', 0),
                    'Training_Time': result.get('training_time', 0)
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Test_F1', ascending=False)
            
            logger.info("\n🏆 Model Rankings (by Test F1-Score):")
            logger.info(comparison_df.round(4).to_string(index=False))
            
            # Best model
            best_model = comparison_df.iloc[0]
            logger.info(f"\n🥇 Best Model: {best_model['Model']}")
            logger.info(f"   • Test F1-Score: {best_model['Test_F1']:.4f}")
            logger.info(f"   • Test Accuracy: {best_model['Test_Accuracy']:.4f}")
            logger.info(f"   • Test ROC-AUC: {best_model['Test_ROC_AUC']:.4f}")
    
    def create_ensemble_model(self, model_names: Optional[List[str]] = None) -> VotingClassifier:
        """
        Create an ensemble model from trained models.
        
        Args:
            model_names: List of model names to include in ensemble
            
        Returns:
            Voting classifier ensemble
        """
        if not model_names:
            # Use top 3 models by performance
            if self.model_results:
                sorted_models = sorted(
                    self.model_results.items(),
                    key=lambda x: x[1].get('test_metrics', {}).get('f1_score', 0),
                    reverse=True
                )
                model_names = [name for name, _ in sorted_models[:3]]
            else:
                model_names = ['RandomForest', 'GradientBoosting', 'LogisticRegression']
        
        logger.info(f"🤝 Creating ensemble with models: {model_names}")
        
        # Prepare estimators
        estimators = []
        for name in model_names:
            if name in self.trained_models:
                estimators.append((name, self.trained_models[name]))
        
        if not estimators:
            logger.error("❌ No trained models available for ensemble")
            return None
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        logger.info(f"✅ Ensemble created with {len(estimators)} models")
        return ensemble
    
    def save_models(self, output_dir: str = 'models') -> None:
        """Save all trained models and results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"💾 Saving {len(self.trained_models)} trained models to {output_dir}/")
        
        # Save individual models
        for name, model in self.trained_models.items():
            model_path = output_path / f'{name.lower()}_model.pkl'
            joblib.dump(model, model_path)
        
        # Save model results
        results_path = output_path / 'model_results.pkl'
        joblib.dump(self.model_results, results_path)
        
        logger.info("✅ All models and results saved")
    
    def load_processed_data(self, data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load preprocessed training and test data."""
        data_path = Path(data_dir)
        
        logger.info(f"📁 Loading processed data from {data_dir}/")
        
        try:
            train_df = pd.read_csv(data_path / 'train_data.csv')
            test_df = pd.read_csv(data_path / 'test_data.csv')
            
            # Separate features and target
            target_col = 'Churn'
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            
            logger.info(f"✅ Data loaded: Train({X_train.shape[0]}, {X_train.shape[1]}), Test({X_test.shape[0]}, {X_test.shape[1]})")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Failed to load processed data: {e}")
            raise

def main():
    """Main execution function."""
    logger.info("🎯 Advanced Model Training for Customer Churn Prediction")
    logger.info("=" * 70)
    
    # Initialize trainer
    trainer = ChurnModelTrainer(random_state=42)
    
    try:
        # Load processed data
        X_train, X_test, y_train, y_test = trainer.load_processed_data('data')
        
        # Train models
        logger.info("\n🚀 Starting model training pipeline...")
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Create ensemble model
        ensemble = trainer.create_ensemble_model()
        if ensemble:
            logger.info("🔄 Training ensemble model...")
            ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = ensemble.predict(X_test)
            y_pred_proba_ensemble = ensemble.predict_proba(X_test)
            
            ensemble_metrics = trainer.evaluator.calculate_metrics(
                y_test, y_pred_ensemble, y_pred_proba_ensemble
            )
            
            logger.info(f"🤝 Ensemble Performance:")
            logger.info(f"   • Accuracy: {ensemble_metrics['accuracy']:.4f}")
            logger.info(f"   • F1-Score: {ensemble_metrics['f1_score']:.4f}")
            logger.info(f"   • ROC-AUC: {ensemble_metrics.get('roc_auc', 0):.4f}")
            
            # Save ensemble
            joblib.dump(ensemble, 'models/ensemble_model.pkl')
        
        # Save all models
        trainer.save_models()
        
        logger.info("\n✅ Model training pipeline completed successfully!")
        logger.info(f"📊 Total models trained: {len(results)}")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()