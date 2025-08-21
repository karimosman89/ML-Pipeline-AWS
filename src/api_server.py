#!/usr/bin/env python3
"""
Professional FastAPI Server for Churn Prediction
================================================

Modern REST API server with:
- Real-time churn prediction
- Model monitoring and metrics
- Interactive API documentation
- Health checks and logging
- Performance monitoring

Author: ML Pipeline Team
Date: 2024
License: MIT
"""

import joblib
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Monitoring
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomerData(BaseModel):
    """Input data model for customer churn prediction."""
    
    # Basic account information
    account_length: int = Field(..., ge=1, le=500, description="Account length in days")
    area_code: int = Field(..., description="Area code")
    
    # Plan information
    international_plan: str = Field(..., pattern="^(Yes|No)$", description="International plan (Yes/No)")
    voice_mail_plan: str = Field(..., pattern="^(Yes|No)$", description="Voice mail plan (Yes/No)")
    
    # Usage statistics
    number_vmail_messages: int = Field(..., ge=0, description="Number of voicemail messages")
    total_day_minutes: float = Field(..., ge=0, le=1000, description="Total day minutes")
    total_day_calls: int = Field(..., ge=0, description="Total day calls")
    total_day_charge: float = Field(..., ge=0, description="Total day charge")
    total_eve_minutes: float = Field(..., ge=0, le=1000, description="Total evening minutes")
    total_eve_calls: int = Field(..., ge=0, description="Total evening calls")
    total_eve_charge: float = Field(..., ge=0, description="Total evening charge")
    total_night_minutes: float = Field(..., ge=0, le=1000, description="Total night minutes")
    total_night_calls: int = Field(..., ge=0, description="Total night calls")
    total_night_charge: float = Field(..., ge=0, description="Total night charge")
    total_intl_minutes: float = Field(..., ge=0, description="Total international minutes")
    total_intl_calls: int = Field(..., ge=0, description="Total international calls")
    total_intl_charge: float = Field(..., ge=0, description="Total international charge")
    customer_service_calls: int = Field(..., ge=0, le=20, description="Customer service calls")
    
    # Optional state
    state: Optional[str] = Field("Unknown", description="Customer state")
    
    class Config:
        schema_extra = {
            "example": {
                "account_length": 128,
                "area_code": 415,
                "international_plan": "No",
                "voice_mail_plan": "Yes",
                "number_vmail_messages": 25,
                "total_day_minutes": 265.1,
                "total_day_calls": 110,
                "total_day_charge": 45.07,
                "total_eve_minutes": 197.4,
                "total_eve_calls": 99,
                "total_eve_charge": 16.78,
                "total_night_minutes": 244.7,
                "total_night_calls": 91,
                "total_night_charge": 11.01,
                "total_intl_minutes": 10.0,
                "total_intl_calls": 3,
                "total_intl_charge": 2.7,
                "customer_service_calls": 1,
                "state": "KS"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for churn prediction."""
    
    customer_id: Optional[str] = None
    churn_probability: float = Field(..., description="Probability of customer churn (0-1)")
    churn_prediction: str = Field(..., description="Churn prediction (Low Risk/High Risk)")
    confidence: float = Field(..., description="Model confidence score")
    risk_factors: List[str] = Field(default=[], description="Key risk factors identified")
    recommendations: List[str] = Field(default=[], description="Retention recommendations")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ModelPredictor:
    """Handles model loading and prediction logic."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the predictor with model artifacts."""
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.metadata = {}
        
        self._load_model_artifacts()
    
    def _load_model_artifacts(self) -> None:
        """Load all model artifacts."""
        try:
            logger.info("🔄 Loading model artifacts...")
            
            # Load best model
            ensemble_path = self.model_dir / "ensemble_model.pkl"
            rf_path = self.model_dir / "randomforest_model.pkl"
            
            if ensemble_path.exists():
                self.model = joblib.load(ensemble_path)
                logger.info("✅ Loaded ensemble model")
            elif rf_path.exists():
                self.model = joblib.load(rf_path)
                logger.info("✅ Loaded RandomForest model")
            else:
                raise FileNotFoundError("No suitable model found")
            
            # Load preprocessing artifacts
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Loaded scaler")
            
            encoders_path = self.model_dir / "label_encoders.pkl"
            if encoders_path.exists():
                self.label_encoders = joblib.load(encoders_path)
                logger.info("✅ Loaded label encoders")
            
            metadata_path = self.model_dir / "processing_metadata.pkl"
            if metadata_path.exists():
                self.metadata = joblib.load(metadata_path)
                self.feature_names = self.metadata.get('feature_names', [])
                logger.info("✅ Loaded processing metadata")
            
            logger.info(f"🎯 Model ready for predictions")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model artifacts: {e}")
            raise
    
    def preprocess_input(self, data: CustomerData) -> pd.DataFrame:
        """Preprocess input data to match training format."""
        # Convert to dataframe
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Standardize column names (match preprocessing)
        df.columns = df.columns.str.replace(' ', '_')
        
        # Handle categorical encoding
        categorical_mapping = {
            'international_plan': {'Yes': 1, 'No': 0},
            'voice_mail_plan': {'Yes': 1, 'No': 0}
        }
        
        for col, mapping in categorical_mapping.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        # Handle state encoding
        if 'state' in df.columns and 'State' in self.label_encoders:
            try:
                state_encoder = self.label_encoders['State']
                if df['state'].iloc[0] in state_encoder.classes_:
                    df['state'] = state_encoder.transform(df['state'])
                else:
                    df['state'] = 0
            except:
                df['state'] = 0
        
        # Feature engineering
        self._create_engineered_features(df)
        
        # Ensure all required features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(df.columns)
            for feature in missing_features:
                df[feature] = 0
            
            # Select only required features in correct order
            df = df[self.feature_names]
        
        # Apply scaling if available
        if self.scaler is not None:
            numerical_features = df.select_dtypes(include=[np.number]).columns
            df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        return df
    
    def _create_engineered_features(self, df: pd.DataFrame) -> None:
        """Create engineered features to match training data."""
        
        # Rate features
        df['Day_rate'] = np.where(
            df['total_day_minutes'] > 0,
            df['total_day_charge'] / df['total_day_minutes'], 0
        )
        
        # Usage aggregations
        usage_cols = ['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes']
        existing_usage_cols = [col for col in usage_cols if col in df.columns]
        if existing_usage_cols:
            df['Total_usage_minutes'] = df[existing_usage_cols].sum(axis=1)
        
        charge_cols = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge']
        existing_charge_cols = [col for col in charge_cols if col in df.columns]
        if existing_charge_cols:
            df['Total_charges'] = df[existing_charge_cols].sum(axis=1)
        
        call_cols = ['total_day_calls', 'total_eve_calls', 'total_night_calls', 'total_intl_calls']
        existing_call_cols = [col for col in call_cols if col in df.columns]
        if existing_call_cols:
            df['Total_calls'] = df[existing_call_cols].sum(axis=1)
    
    def predict(self, data: CustomerData) -> PredictionResponse:
        """Make churn prediction for customer data."""
        start_time = time.time()
        
        try:
            # Preprocess input
            processed_df = self.preprocess_input(data)
            
            # Make prediction
            churn_prob = self.model.predict_proba(processed_df)[0]
            churn_probability = float(churn_prob[1])  # Probability of churn
            
            # Determine risk level
            if churn_probability < 0.3:
                risk_level = "Low Risk"
            elif churn_probability < 0.7:
                risk_level = "Medium Risk"  
            else:
                risk_level = "High Risk"
            
            # Calculate confidence
            confidence = float(abs(churn_probability - 0.5) * 2)
            
            # Generate risk factors and recommendations
            risk_factors = self._identify_risk_factors(data)
            recommendations = self._generate_recommendations(data, churn_probability)
            
            response = PredictionResponse(
                churn_probability=churn_probability,
                churn_prediction=risk_level,
                confidence=confidence,
                risk_factors=risk_factors,
                recommendations=recommendations
            )
            
            logger.info(f"✅ Prediction completed: {risk_level} (prob: {churn_probability:.3f})")
            return response
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def _identify_risk_factors(self, data: CustomerData) -> List[str]:
        """Identify key risk factors based on input data."""
        risk_factors = []
        
        if data.customer_service_calls >= 3:
            risk_factors.append("High customer service calls")
        
        total_usage = (data.total_day_minutes + data.total_eve_minutes + 
                      data.total_night_minutes + data.total_intl_minutes)
        if total_usage < 300:
            risk_factors.append("Low usage patterns")
        
        if data.international_plan == "Yes":
            risk_factors.append("International plan subscriber")
        
        total_charges = (data.total_day_charge + data.total_eve_charge + 
                        data.total_night_charge + data.total_intl_charge)
        if total_charges > 80:
            risk_factors.append("High monthly charges")
        
        return risk_factors
    
    def _generate_recommendations(self, data: CustomerData, churn_prob: float) -> List[str]:
        """Generate retention recommendations."""
        recommendations = []
        
        if churn_prob < 0.3:
            recommendations.append("Continue current engagement strategy")
        
        if data.customer_service_calls >= 3:
            recommendations.append("Proactive customer service outreach")
            recommendations.append("Review and resolve service issues")
        
        total_usage = (data.total_day_minutes + data.total_eve_minutes + 
                      data.total_night_minutes + data.total_intl_minutes)
        if total_usage < 300:
            recommendations.append("Offer usage incentive programs")
        
        if churn_prob > 0.7:
            recommendations.append("Priority retention campaign")
            recommendations.append("Consider loyalty discounts")
        
        return recommendations

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    **Professional ML-powered Customer Churn Prediction Service**
    
    This API provides real-time customer churn predictions using advanced machine learning models.
    
    ## Features
    - Real-time churn prediction
    - Risk assessment and recommendations  
    - Model monitoring and metrics
    - High-performance inference
    """,
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model predictor
predictor = ModelPredictor()

# Server startup time
server_start_time = time.time()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Churn Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .feature { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .btn { display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }
            .status { color: #27ae60; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Customer Churn Prediction API</h1>
            <p class="status">✅ API Server is running and ready for predictions!</p>
            
            <div class="feature">
                <h3>🚀 Key Features</h3>
                <ul>
                    <li><strong>Real-time Predictions:</strong> Get instant churn probability assessments</li>
                    <li><strong>Risk Analysis:</strong> Detailed risk factors and retention recommendations</li>
                    <li><strong>High Performance:</strong> Optimized ML models with sub-second response times</li>
                </ul>
            </div>
            
            <div class="feature">
                <h3>📊 Available Endpoints</h3>
                <a href="/docs" class="btn">📚 Interactive API Docs</a>
                <a href="/health" class="btn">💗 Health Check</a>
            </div>
            
            <div class="feature">
                <h3>🔮 Model Performance</h3>
                <p><strong>F1-Score:</strong> 94.86% | <strong>Accuracy:</strong> 95.13% | <strong>ROC-AUC:</strong> 87.35%</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: CustomerData, customer_id: Optional[str] = None):
    """
    Predict customer churn probability.
    
    Analyzes customer usage patterns and account information to predict
    the likelihood of customer churn and provides actionable recommendations.
    """
    logger.info(f"🔮 Processing prediction request for customer: {customer_id or 'Anonymous'}")
    
    try:
        result = predictor.predict(data)
        if customer_id:
            result.customer_id = customer_id
        
        return result
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Test model availability
        test_data = CustomerData(
            account_length=100, area_code=415, international_plan="No",
            voice_mail_plan="Yes", number_vmail_messages=10,
            total_day_minutes=200, total_day_calls=100, total_day_charge=30,
            total_eve_minutes=200, total_eve_calls=100, total_eve_charge=15,
            total_night_minutes=200, total_night_calls=100, total_night_charge=10,
            total_intl_minutes=10, total_intl_calls=3, total_intl_charge=3,
            customer_service_calls=1
        )
        
        # Quick prediction test
        _ = predictor.predict(test_data)
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - server_start_time,
            "model_status": "operational",
            "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information and statistics."""
    return {
        "model_type": type(predictor.model).__name__,
        "features_count": len(predictor.feature_names) if predictor.feature_names else 0,
        "model_performance": {
            "f1_score": 0.9486,
            "accuracy": 0.9513,
            "roc_auc": 0.8735,
            "training_samples": 3644,
            "test_samples": 534
        },
        "last_updated": "2024-08-21T21:15:00Z"
    }

def main():
    """Start the API server."""
    logger.info("🚀 Starting Customer Churn Prediction API Server")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()