"""
Scoring Script for Azure ML Managed Endpoint
This script handles inference requests for the dynamic pricing model
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
feature_columns = [
    "base_price", "cost", "competitor_price", "demand", "inventory_level",
    "customer_engagement", "market_demand_factor", "seasonal_factor",
    "price_to_cost_ratio", "profit_margin", "is_profitable", "price_vs_competitor",
    "price_change", "demand_change", "price_elasticity", "demand_trend_7d",
    "price_volatility_7d", "avg_price_30d", "inventory_velocity",
    "competitive_position", "revenue_per_unit", "profit_per_unit",
    "day_of_week", "month", "quarter", "is_weekend", "is_month_end",
    "category_avg_price", "category_price_rank", "price_vs_category_avg",
    "demand_supply_ratio", "profit_optimization_score", "market_positioning_score"
]

def init():
    """
    Initialize the model when the container starts
    """
    global model
    try:
        # Get the path to the registered model
        model_path = os.getenv("AZUREML_MODEL_DIR", "./model")
        model_file = os.path.join(model_path, "model.pkl")
        
        # Load the model
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            logger.info(f"✅ Model loaded successfully from {model_file}")
        else:
            # Fallback: try to load MLflow model
            import mlflow
            model = mlflow.pyfunc.load_model(model_path)
            logger.info(f"✅ MLflow model loaded successfully from {model_path}")
        
        logger.info(f"🎯 Model type: {type(model)}")
        logger.info(f"📊 Expected features: {len(feature_columns)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def run(raw_data):
    """
    Handle inference requests
    
    Args:
        raw_data: JSON string containing input data
        
    Returns:
        JSON response with prediction and metadata
    """
    try:
        # Parse input data
        logger.info(f"📥 Received request: {raw_data[:200]}...")
        data = json.loads(raw_data)
        
        # Handle different input formats
        if isinstance(data, dict):
            # Single prediction
            input_data = [data]
        elif isinstance(data, list):
            # Batch prediction
            input_data = data
        else:
            raise ValueError("Input must be a dictionary or list of dictionaries")
        
        predictions = []
        
        for item in input_data:
            # Validate required features
            missing_features = [col for col in feature_columns if col not in item]
            if missing_features:
                return {
                    "error": f"Missing required features: {missing_features[:10]}...",
                    "required_features": feature_columns,
                    "provided_features": list(item.keys()),
                    "missing_count": len(missing_features)
                }
            
            # Prepare input array
            input_array = np.array([[item[col] for col in feature_columns]])
            
            # Make prediction
            if hasattr(model, 'predict'):
                # Scikit-learn or similar model
                prediction = float(model.predict(input_array)[0])
            else:
                # MLflow model
                input_df = pd.DataFrame([item])
                prediction = float(model.predict(input_df)[0])
            
            # Calculate confidence interval (simplified)
            confidence_range = prediction * 0.05  # 5% range
            confidence_interval = [
                round(prediction - confidence_range, 2),
                round(prediction + confidence_range, 2)
            ]
            
            # Create response
            result = {
                "predicted_price": round(prediction, 2),
                "confidence_interval": confidence_interval,
                "input_features": {
                    "base_price": item.get("base_price"),
                    "competitor_price": item.get("competitor_price"),
                    "demand": item.get("demand"),
                    "inventory_level": item.get("inventory_level")
                },
                "business_metrics": {
                    "profit_margin": round((prediction - item.get("cost", 0)) / prediction * 100, 2) if prediction > 0 else 0,
                    "competitive_position": round(prediction / item.get("competitor_price", 1), 3) if item.get("competitor_price", 0) > 0 else 0
                },
                "model_metadata": {
                    "model_name": "dynamic_pricing_model",
                    "model_version": os.getenv("MODEL_VERSION", "1"),
                    "features_used": len(feature_columns),
                    "prediction_timestamp": datetime.now().isoformat()
                }
            }
            
            predictions.append(result)
        
        # Return single prediction or batch predictions
        if len(predictions) == 1:
            response = predictions[0]
        else:
            response = {
                "predictions": predictions,
                "batch_size": len(predictions),
                "processing_timestamp": datetime.now().isoformat()
            }
        
        logger.info(f"✅ Prediction successful: {prediction:.2f}")
        return response
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing error: {e}")
        return {
            "error": "Invalid JSON format",
            "details": str(e),
            "example_format": {
                "base_price": 100.0,
                "cost": 60.0,
                "competitor_price": 95.0,
                "demand": 150
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {
            "error": "Prediction failed",
            "details": str(e),
            "model_info": {
                "model_loaded": model is not None,
                "expected_features": len(feature_columns)
            }
        }

def get_model_info():
    """
    Get information about the loaded model
    """
    return {
        "model_type": str(type(model)),
        "feature_count": len(feature_columns),
        "feature_names": feature_columns[:10],  # Show first 10
        "model_loaded": model is not None,
        "scoring_script_version": "1.0.0"
    }

# Health check endpoint
def health_check():
    """
    Health check for the endpoint
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
