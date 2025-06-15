"""
Configuration constants and settings for the unified dynamic pricing pipeline
"""

from typing import Dict, List, Any, Tuple

# ===== DATA VALIDATION CONFIGURATION =====
DEFAULT_VALIDATION_RULES = {
    'required_columns': ['price', 'demand', 'cost', 'competitor_price', 'date'],
    'numeric_columns': ['price', 'demand', 'cost', 'competitor_price'],
    'date_columns': ['date'],
    'price_range': (0, 10000),
    'demand_range': (0, float('inf')),
    'cost_range': (0, float('inf')),
    'quality_thresholds': {
        'excellent': 90,
        'good': 75,
        'acceptable': 60,
        'poor': 40
    }
}

# ===== COLUMN DETECTION PATTERNS =====
COLUMN_PATTERNS = {
    'price_patterns': ['price', 'selling', 'mrp', 'cost_price', 'unit_price'],
    'demand_patterns': ['demand', 'units', 'sold', 'quantity', 'volume'],
    'cost_patterns': ['cost', 'cogs', 'expense', 'base_cost'],
    'date_patterns': ['date', 'time', 'timestamp', 'created', 'updated'],
    'competitor_patterns': ['competitor', 'base', 'benchmark', 'market_price'],
    'inventory_patterns': ['stock', 'inventory', 'available', 'onhand'],
    'customer_patterns': ['ctr', 'bounce', 'session', 'conversion', 'engagement']
}

# ===== FEATURE ENGINEERING CONFIGURATION =====
FEATURE_WINDOWS = {
    'short_term': [3, 7],
    'medium_term': [14, 21, 30],
    'long_term': [60, 90, 180]
}

FEATURE_CONFIG = {
    'pricing_elasticity': {
        'elasticity_window': 7,
        'volatility_windows': [7, 14, 30],
        'trend_windows': [7, 30]
    },
    'customer_behavior': {
        'ltv_window': 30,
        'engagement_metrics': ['ctr', 'bounce_rate', 'session_duration'],
        'conversion_window': 14
    },
    'inventory_optimization': {
        'inventory_window': 14,
        'stockout_threshold': 0.95,
        'turnover_windows': [7, 14, 30]
    },
    'time_series': {
        'lag_periods': [1, 3, 7, 14],
        'rolling_windows': [3, 7, 14, 30],
        'momentum_periods': [3, 7]
    },
    'seasonal': {
        'holiday_months': [11, 12],
        'peak_months': [11, 12, 1],
        'summer_months': [6, 7, 8]
    }
}

# ===== MODEL TRAINING CONFIGURATION =====
MODEL_DEFAULTS = {
    'cv_folds': 5,
    'scoring': 'r2',
    'random_state': 42,
    'n_jobs': -1,
    'test_size': 0.2,
    'validation_split': 0.2
}

# Model-specific hyperparameters
MODEL_HYPERPARAMETERS = {
    'ridge': {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    'lasso': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
    },
    'elastic_net': {
        'alpha': [0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    'lightgbm': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 50]
    }
}

# ===== DATA PROCESSING CONFIGURATION =====
PROCESSING_CONFIG = {
    'missing_values': {
        'strategies': ['auto', 'median', 'mean', 'mode', 'knn', 'forward_fill'],
        'knn_neighbors': 5,
        'threshold_for_knn': 20,  # percentage
        'threshold_for_ffill': 50  # percentage
    },
    'outliers': {
        'methods': ['iqr', 'zscore', 'percentile'],
        'iqr_multiplier': 1.5,
        'zscore_threshold': 3,
        'percentile_bounds': (0.05, 0.95)
    },
    'encoding': {
        'low_cardinality_threshold': 10,
        'medium_cardinality_threshold': 50,
        'high_cardinality_warning': True
    },
    'scaling': {
        'methods': ['standard', 'robust', 'minmax'],
        'default_method': 'robust'
    }
}

# ===== BUSINESS RULES CONFIGURATION =====
BUSINESS_RULES = {
    'pricing': {
        'min_price': 0,
        'max_price': 100000,
        'min_margin_percentage': 5,
        'max_discount_percentage': 80
    },
    'demand': {
        'min_demand': 0,
        'max_reasonable_demand': 10000
    },
    'inventory': {
        'min_stock_level': 0,
        'stockout_threshold': 0.95,
        'overstock_multiplier': 10
    }
}

# ===== EVALUATION METRICS CONFIGURATION =====
EVALUATION_CONFIG = {
    'primary_metrics': ['r2', 'rmse', 'mae'],
    'business_metrics': ['price_accuracy_5pct', 'price_accuracy_10pct'],
    'thresholds': {
        'excellent_r2': 0.9,
        'good_r2': 0.8,
        'acceptable_r2': 0.7,
        'price_accuracy_target': 0.9  # 90% within 10%
    }
}

# ===== LOGGING CONFIGURATION =====
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'log_to_file': False,
    'log_file': 'unified_pipeline.log'
}

# ===== PERFORMANCE CONFIGURATION =====
PERFORMANCE_CONFIG = {
    'memory_optimization': True,
    'chunk_size': 10000,
    'parallel_processing': True,
    'cache_transformations': True,
    'early_stopping': True
}

# ===== PIPELINE DEFAULTS =====
DEFAULT_PIPELINE_CONFIG = {
    'data_processor': {
        'missing_value_strategy': 'auto',
        'outlier_method': 'iqr',
        'scaling_method': 'robust',
        'encoding_threshold': 10
    },
    'feature_engineer': {
        'elasticity_window': 7,
        'ltv_window': 30,
        'inventory_window': 14,
        'create_lag_features': True,
        'create_seasonal_features': True
    },
    'model_trainer': {
        'cv_folds': 5,
        'scoring': 'r2',
        'enable_hyperparameter_tuning': True,
        'early_stopping': True
    }
}

# ===== VERSION AND METADATA =====
PIPELINE_METADATA = {
    'version': '1.0.0',
    'name': 'Unified Dynamic Pricing Pipeline',
    'description': 'Consolidated ML pipeline for dynamic pricing optimization',
    'author': 'Dynamic Pricing Team',
    'created_date': '2024',
    'last_updated': '2024'
}


def get_default_config() -> Dict[str, Any]:
    """Get the default pipeline configuration"""
    return DEFAULT_PIPELINE_CONFIG.copy()


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model"""
    if model_name not in MODEL_HYPERPARAMETERS:
        raise ValueError(f"Model {model_name} not found in configuration")
    return MODEL_HYPERPARAMETERS[model_name].copy()


def get_feature_config(feature_type: str) -> Dict[str, Any]:
    """Get configuration for a specific feature type"""
    if feature_type not in FEATURE_CONFIG:
        raise ValueError(f"Feature type {feature_type} not found in configuration")
    return FEATURE_CONFIG[feature_type].copy()


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate pipeline configuration"""
    required_sections = ['data_processor', 'feature_engineer', 'model_trainer']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section '{section}' missing")
    
    return True


def merge_configs(base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge custom configuration with base configuration"""
    merged = base_config.copy()
    
    for key, value in custom_config.items():
        if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key].update(value)
        else:
            merged[key] = value
    
    return merged
