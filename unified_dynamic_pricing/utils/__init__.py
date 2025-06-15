"""
Utility functions for unified dynamic pricing pipeline
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def create_sample_pricing_data(n_days: int = 365, n_products: int = 5) -> pd.DataFrame:
    """Create realistic sample pricing data for testing"""
    np.random.seed(42)
    
    # Base parameters
    products = [f"Product_{i+1}" for i in range(n_products)]
    base_prices = np.random.uniform(50, 500, n_products)
    base_costs = base_prices * np.random.uniform(0.4, 0.7, n_products)
    
    data = []
    
    for day in range(n_days):
        date = datetime.now() - timedelta(days=n_days-day-1)
        
        # Seasonal factors
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day / 365)
        weekend_factor = 1.1 if date.weekday() >= 5 else 1.0
        holiday_factor = 1.3 if date.month == 12 else 1.0
        
        for i, product in enumerate(products):
            # Price with some randomness and trends
            price_trend = 1 + 0.001 * day  # Slight upward trend
            price_noise = np.random.normal(1, 0.05)
            current_price = base_prices[i] * seasonal_factor * weekend_factor * holiday_factor * price_trend * price_noise
            
            # Competitor price (usually within 10% of our price)
            competitor_price = current_price * np.random.uniform(0.9, 1.1)
            
            # Cost with slight variation
            current_cost = base_costs[i] * np.random.uniform(0.95, 1.05)
            
            # Demand based on price elasticity and external factors
            price_elasticity = -1.5  # Demand decreases as price increases
            base_demand = 100
            demand = base_demand * (current_price / base_prices[i]) ** price_elasticity
            demand *= seasonal_factor * weekend_factor * holiday_factor
            demand *= np.random.uniform(0.8, 1.2)  # Random variation
            demand = max(0, int(demand))
            
            # Units sold (fulfilled demand with some stockouts)
            stockout_probability = max(0, (demand - 150) / 200)  # Higher demand = higher stockout risk
            if np.random.random() < stockout_probability:
                units_sold = int(demand * np.random.uniform(0.7, 0.9))
            else:
                units_sold = demand
            
            # Customer behavior metrics
            ctr = np.random.uniform(0.02, 0.08)
            bounce_rate = np.random.uniform(0.3, 0.7)
            abandoned_cart_rate = np.random.uniform(0.2, 0.6)
            session_duration = np.random.uniform(120, 600)  # seconds
            
            # Stock levels
            stock_start = np.random.randint(100, 300)
            stock_end = max(0, stock_start - units_sold)
            
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Product': product,
                'SellingPrice': round(current_price, 2),
                'Cost': round(current_cost, 2),
                'CompetitorPrice': round(competitor_price, 2),
                'Demand': demand,
                'UnitsSold': units_sold,
                'DemandFulfilled': units_sold,
                'StockStart': stock_start,
                'StockEnd': stock_end,
                'CTR': round(ctr, 4),
                'BounceRate': round(bounce_rate, 3),
                'AbandonedCartRate': round(abandoned_cart_rate, 3),
                'AvgSessionDuration_sec': int(session_duration),
                'Backorders': max(0, demand - units_sold)
            })
    
    df = pd.DataFrame(data)
    logger.info(f"Created sample pricing data with {len(df)} records")
    return df


def print_pipeline_results(results: Dict[str, Any]):
    """Print formatted pipeline results"""
    print("\n" + "="*80)
    print("UNIFIED DYNAMIC PRICING PIPELINE RESULTS")
    print("="*80)
    
    if results.get('pipeline_status') == 'failed':
        print(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        return
    
    # Data validation results
    print("\n📊 DATA VALIDATION RESULTS")
    print("-" * 40)
    validation = results.get('data_validation', {})
    print(f"Schema Valid: {'✅' if validation.get('schema_valid') else '❌'}")
    print(f"Quality Score: {validation.get('quality_score', 0):.1f}/100")
    print(f"Overall Valid: {'✅' if validation.get('overall_valid') else '❌'}")
    
    # Data shapes
    print("\n📈 DATA PIPELINE SHAPES")
    print("-" * 40)
    shapes = results.get('data_shapes', {})
    for step, shape in shapes.items():
        print(f"{step.title()}: {shape}")
    
    # Model training results
    print("\n🤖 MODEL TRAINING RESULTS")
    print("-" * 40)
    best_model = results.get('best_model')
    if best_model:
        print(f"Best Model: {best_model['name']}")
        print(f"CV Score (R²): {best_model['cv_score']:.4f}")
        print(f"Best Parameters: {best_model.get('params', {})}")
    
    # Model comparison
    model_comparison = results.get('model_comparison', [])
    if model_comparison:
        print(f"\nTrained {len(model_comparison)} models:")
        for model in model_comparison[:5]:  # Show top 5
            print(f"  • {model['Model']}: R² = {model['CV_Score_R2']:.4f}")
    
    # Feature engineering summary
    feature_summary = results.get('feature_engineering_summary', {})
    if feature_summary:
        print(f"\n🔧 FEATURE ENGINEERING")
        print("-" * 40)
        print(f"Total Features Created: {feature_summary.get('total_created_features', 0)}")
        categories = feature_summary.get('feature_categories', {})
        for category, count in categories.items():
            if count > 0:
                print(f"  • {category.replace('_', ' ').title()}: {count}")
    
    # Feature importance
    feature_importance = results.get('feature_importance', [])
    if feature_importance:
        print(f"\n⭐ TOP FEATURE IMPORTANCE")
        print("-" * 40)
        for i, feature in enumerate(feature_importance[:10]):
            print(f"{i+1:2d}. {feature['Feature']}: {feature['Importance']:.4f}")
    
    print("\n" + "="*80)


def print_feature_summary():
    """Print summary of unified pipeline features"""
    print("\n" + "="*80)
    print("UNIFIED DYNAMIC PRICING PIPELINE FEATURES")
    print("="*80)
    
    features = {
        "📊 Data Processing & Validation": [
            "• Comprehensive data quality assessment with scoring",
            "• Intelligent missing value handling (KNN, median, mode)",
            "• Business-rule based outlier detection and treatment",
            "• Smart categorical encoding based on cardinality",
            "• Automated temporal feature extraction",
            "• Robust feature scaling with multiple methods"
        ],
        
        "🎯 Advanced Feature Engineering": [
            "• Pricing elasticity and demand sensitivity features",
            "• Customer behavior and engagement scoring",
            "• Inventory optimization and stock management features",
            "• Time series features (lags, rolling stats, momentum)",
            "• Seasonal and calendar-based features",
            "• Competitive positioning and profit margin analysis"
        ],
        
        "🤖 ML Model Training & Optimization": [
            "• Multiple algorithms: Linear, Tree-based, Gradient Boosting",
            "• Advanced models: XGBoost, LightGBM (when available)",
            "• Automated hyperparameter optimization with GridSearchCV",
            "• Time series aware cross-validation with TimeSeriesSplit",
            "• Comprehensive evaluation metrics (RMSE, R², MAPE, Business metrics)",
            "• Automated model selection and comparison"
        ],
        
        "⚡ Pipeline Integration & Usability": [
            "• Unified API combining Module 1 and Module 2 functionality",
            "• Flexible data loading from files, DataFrames, or multiple sources",
            "• Automated target column detection",
            "• Built-in prediction capabilities for new data",
            "• Pipeline state management with save/load functionality",
            "• Comprehensive logging and error handling"
        ],
        
        "📈 Business Intelligence Features": [
            "• Price accuracy metrics (within 5% and 10% thresholds)",
            "• Revenue optimization features",
            "• Customer lifetime value approximation",
            "• Inventory turnover and service level metrics",
            "• Competitive analysis and market positioning",
            "• Seasonal demand forecasting capabilities"
        ]
    }
    
    for category, feature_list in features.items():
        print(f"\n{category}")
        print("-" * 60)
        for feature in feature_list:
            print(feature)
    
    print("\n" + "="*80)


def generate_qa_answers():
    """Generate Q&A answers for Module 2 requirements"""
    qa_answers = {
        "Q1: Differences between MSE, RMSE, and R² as evaluation metrics": {
            "answer": """
**MSE (Mean Squared Error)**:
- Measures average squared differences between actual and predicted values
- Units are squared, making interpretation difficult
- Heavily penalizes large errors due to squaring
- Range: 0 to ∞ (lower is better)

**RMSE (Root Mean Squared Error)**:
- Square root of MSE, returning to original units
- More interpretable than MSE
- Still penalizes large errors more than small ones
- Same units as the target variable

**R² (Coefficient of Determination)**:
- Represents proportion of variance in target explained by model
- Scale-independent (0 to 1, where 1 is perfect)
- Easy to interpret as percentage of variance explained
- Allows comparison across different datasets and models

**Usage in Pricing**:
- Use RMSE when you need interpretable error in dollar terms
- Use R² for model comparison and explaining variance
- Use MSE when mathematical properties of squared errors are needed
            """,
            "implementation": "All three metrics are implemented in our unified model trainer with comprehensive business-specific pricing accuracy metrics."
        },
        
        "Q2: How regression models with regularization help prevent overfitting": {
            "answer": """
**Regularization Techniques**:

**Ridge (L2) Regularization**:
- Adds penalty term: λ * Σ(βi²) to cost function
- Shrinks coefficients toward zero but doesn't eliminate them
- Handles multicollinearity well
- Good when all features are somewhat relevant

**Lasso (L1) Regularization**:
- Adds penalty term: λ * Σ|βi| to cost function
- Can set coefficients to exactly zero (feature selection)
- Performs automatic feature selection
- Good when many features are irrelevant

**Elastic Net**:
- Combines L1 and L2: λ₁ * Σ|βi| + λ₂ * Σ(βi²)
- Balances feature selection and coefficient shrinkage
- Handles groups of correlated features better than Lasso

**Overfitting Prevention**:
- Constrains model complexity by limiting coefficient magnitudes
- Reduces model's sensitivity to small changes in training data
- Improves generalization to unseen data
- Particularly effective with high-dimensional feature spaces
            """,
            "implementation": "Our unified pipeline implements Ridge, Lasso, and Elastic Net with automated hyperparameter tuning to find optimal regularization strength."
        }
    }
    
    return qa_answers


def print_qa_answers():
    """Print formatted Q&A answers"""
    print("\n" + "="*80)
    print("MODULE 2 Q&A ANSWERS")
    print("="*80)
    
    qa_answers = generate_qa_answers()
    
    for question, content in qa_answers.items():
        print(f"\n{question}")
        print("-" * len(question))
        print(content["answer"])
        print(f"\n💡 Implementation: {content['implementation']}")
        print()
    
    print("="*80)


def create_pipeline_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create optimized pipeline configuration"""
    default_config = {
        'data_processor': {
            'missing_value_strategy': 'auto',
            'outlier_method': 'iqr',
            'scaling_method': 'robust'
        },
        'feature_engineer': {
            'elasticity_window': 7,
            'ltv_window': 30,
            'inventory_window': 14
        },
        'model_trainer': {
            'cv_folds': 5,
            'scoring': 'r2',
            'n_jobs': -1
        }
    }
    
    if custom_config:
        # Deep merge custom config
        for key, value in custom_config.items():
            if key in default_config and isinstance(value, dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return default_config
