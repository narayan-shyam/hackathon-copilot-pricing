"""
Unified Model Training Pipeline
Consolidated ML model training with multiple algorithms and hyperparameter optimization
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
import datetime

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Check for advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available - install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available - install with: pip install lightgbm")

# MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
    logger.info("MLflow integration enabled")
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available - install with: pip install mlflow")


class UnifiedModelTrainer:
    """Comprehensive ML model training with multiple algorithms and optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.models = {}
        self.trained_models = {}
        self.model_performance = {}
        self.best_model = None
        self.feature_importance = {}
        
        # MLflow configuration
        self.mlflow_enabled = MLFLOW_AVAILABLE and self.config.get('enable_mlflow', True)
        self.experiment_name = self.config.get('experiment_name', 'dynamic_pricing_pipeline')
        
        if self.mlflow_enabled:
            self.setup_mlflow()
            
    def setup_mlflow(self):
        """Setup MLflow experiment tracking"""
        try:
            # Set or create experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(self.experiment_name)
                    logger.info(f"Created MLflow experiment: {self.experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
            except:
                # Fallback to setting experiment by name
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"Set MLflow experiment: {self.experiment_name}")
                
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}. Continuing without MLflow tracking.")
            self.mlflow_enabled = False
            
    def log_to_mlflow(self, model_name: str, model: Any, params: Dict, metrics: Dict, 
                     feature_importance: Dict = None):
        """Log model results to MLflow"""
        if not self.mlflow_enabled:
            return
            
        try:
            with mlflow.start_run(run_name=f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                if 'xgb' in model_name.lower() and XGBOOST_AVAILABLE:
                    mlflow.xgboost.log_model(model, "model")
                elif 'lightgbm' in model_name.lower() and LIGHTGBM_AVAILABLE:
                    mlflow.lightgbm.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                
                # Log feature importance as artifact
                if feature_importance:
                    importance_df = pd.DataFrame([
                        {'feature': k, 'importance': v} for k, v in feature_importance.items()
                    ]).sort_values('importance', ascending=False)
                    
                    importance_df.to_csv('feature_importance.csv', index=False)
                    mlflow.log_artifact('feature_importance.csv')
                
                # Log model metadata
                mlflow.set_tag("model_type", type(model).__name__)
                mlflow.set_tag("framework", "scikit-learn")
                
                logger.info(f"Logged {model_name} to MLflow")
                
        except Exception as e:
            logger.warning(f"Failed to log {model_name} to MLflow: {e}")
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize comprehensive set of ML models with parameter grids"""
        
        # Linear models with regularization
        self.models['linear_regression'] = {
            'model': LinearRegression(),
            'params': {},
            'description': 'Basic linear regression'
        }
        
        self.models['ridge'] = {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'description': 'Ridge regression with L2 regularization'
        }
        
        self.models['lasso'] = {
            'model': Lasso(random_state=42, max_iter=2000),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'description': 'Lasso regression with L1 regularization'
        }
        
        self.models['elastic_net'] = {
            'model': ElasticNet(random_state=42, max_iter=2000),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'description': 'Elastic Net with L1 and L2 regularization'
        }
        
        # Tree-based ensemble models
        self.models['random_forest'] = {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'description': 'Random Forest ensemble'
        }
        
        self.models['gradient_boosting'] = {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'description': 'Gradient Boosting Machine'
        }
        
        self.models['extra_trees'] = {
            'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'description': 'Extremely Randomized Trees'
        }
        
        # Advanced gradient boosting (if available)
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = {
                'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                },
                'description': 'XGBoost Gradient Boosting'
            }
        
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = {
                'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 50]
                },
                'description': 'LightGBM Gradient Boosting'
            }
        
        logger.info(f"Initialized {len(self.models)} models for training")
        return self.models
    
    def calculate_comprehensive_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics for regression"""
        
        # Basic statistical metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Handle edge cases for MAPE
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Additional metrics
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Business-specific metrics for pricing
        relative_errors = np.abs(residuals / (y_true + 1e-8))
        price_accuracy_5pct = np.mean(relative_errors <= 0.05) * 100  # Within 5%
        price_accuracy_10pct = np.mean(relative_errors <= 0.10) * 100  # Within 10%
        
        # Adjusted R-squared (if we have the number of features)
        n = len(y_true)
        p = 1  # Default to 1 if we don't know the number of features
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'adj_r2': adj_r2,
            'mape': mape,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'price_accuracy_5pct': price_accuracy_5pct,
            'price_accuracy_10pct': price_accuracy_10pct
        }
    
    def train_model_with_optimization(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                                    cv_folds: int = 5) -> Dict[str, Any]:
        """Train single model with cross-validation and hyperparameter optimization"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_config = self.models[model_name]
        base_model = model_config['model']
        param_grid = model_config['params']
        
        logger.info(f"Training {model_name} with hyperparameter optimization")
        
        try:
            # Use TimeSeriesSplit for time series data
            cv_strategy = TimeSeriesSplit(n_splits=cv_folds)
            
            if param_grid:
                # Hyperparameter optimization with GridSearchCV
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv_strategy,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_score = grid_search.best_score_
                
            else:
                # Simple cross-validation without hyperparameter tuning
                cv_scores = cross_val_score(base_model, X, y, cv=cv_strategy, scoring='r2')
                best_model = base_model.fit(X, y)
                best_params = {}
                cv_score = cv_scores.mean()
            
            # Feature importance
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, best_model.feature_importances_))
            elif hasattr(best_model, 'coef_'):
                feature_importance = dict(zip(X.columns, np.abs(best_model.coef_)))
            
            # Store results
            results = {
                'model': best_model,
                'best_params': best_params,
                'cv_score': cv_score,
                'feature_importance': feature_importance,
                'description': model_config['description']
            }
            
            self.trained_models[model_name] = results
            
            # Log to MLflow
            if self.mlflow_enabled:
                mlflow_metrics = {'cv_r2_score': cv_score}
                self.log_to_mlflow(
                    model_name=model_name,
                    model=best_model,
                    params=best_params,
                    metrics=mlflow_metrics,
                    feature_importance=feature_importance
                )
            
            logger.info(f"Model {model_name} trained successfully. CV Score: {cv_score:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to train model {model_name}: {str(e)}")
            return {}
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """Train all models with cross-validation and hyperparameter optimization"""
        
        if not self.models:
            self.initialize_models()
        
        logger.info(f"Starting training for {len(self.models)} models")
        
        training_results = {}
        
        for model_name in self.models.keys():
            try:
                results = self.train_model_with_optimization(model_name, X, y, cv_folds)
                if results:  # Only store successful results
                    training_results[model_name] = results
                    self.model_performance[model_name] = {
                        'cv_score': results['cv_score'],
                        'model_type': results['description']
                    }
                    
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        # Select best model based on CV score
        if self.model_performance:
            best_model_name = max(self.model_performance.keys(), 
                                key=lambda x: self.model_performance[x]['cv_score'])
            self.best_model = {
                'name': best_model_name,
                'model': self.trained_models[best_model_name]['model'],
                'cv_score': self.model_performance[best_model_name]['cv_score'],
                'params': self.trained_models[best_model_name]['best_params']
            }
            
            logger.info(f"Best model: {best_model_name} (CV Score: {self.best_model['cv_score']:.4f})")
        
        return training_results
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Comprehensive evaluation of all trained models"""
        
        if not self.trained_models:
            raise ValueError("No trained models found. Please train models first.")
        
        logger.info("Starting comprehensive model evaluation")
        
        evaluation_results = {}
        
        for model_name, model_info in self.trained_models.items():
            try:
                model = model_info['model']
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate comprehensive metrics
                metrics = self.calculate_comprehensive_metrics(y_test, y_pred)
                
                evaluation_results[model_name] = {
                    'metrics': metrics,
                    'cv_score': model_info['cv_score'],
                    'best_params': model_info['best_params'],
                    'description': model_info['description']
                }
                
                # Log evaluation metrics to MLflow
                if self.mlflow_enabled:
                    eval_metrics = {
                        'test_r2': metrics['r2'],
                        'test_rmse': metrics['rmse'],
                        'test_mae': metrics['mae'],
                        'test_mape': metrics['mape'],
                        'price_accuracy_5pct': metrics['price_accuracy_5pct'],
                        'price_accuracy_10pct': metrics['price_accuracy_10pct']
                    }
                    
                    # Update the existing MLflow run with evaluation metrics
                    try:
                        with mlflow.start_run(run_name=f"{model_name}_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                            mlflow.log_metrics(eval_metrics)
                            mlflow.log_params(model_info['best_params'])
                            mlflow.set_tag("stage", "evaluation")
                    except Exception as e:
                        logger.warning(f"Failed to log evaluation metrics to MLflow: {e}")
                
                logger.info(f"Model {model_name} - Test R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate model {model_name}: {str(e)}")
                continue
        
        return evaluation_results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get detailed comparison of all trained models"""
        if not self.model_performance:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, performance in self.model_performance.items():
            model_info = self.trained_models[model_name]
            comparison_data.append({
                'Model': model_name,
                'CV_Score_R2': performance['cv_score'],
                'Model_Type': performance['model_type'],
                'Best_Params': str(model_info['best_params'])[:100] + '...' if len(str(model_info['best_params'])) > 100 else str(model_info['best_params'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('CV_Score_R2', ascending=False)
        return comparison_df
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from the best model or aggregate from all models"""
        
        if not self.trained_models:
            return pd.DataFrame()
        
        # Get feature importance from best model if available
        if self.best_model and self.best_model['name'] in self.trained_models:
            best_model_info = self.trained_models[self.best_model['name']]
            if best_model_info['feature_importance']:
                importance_df = pd.DataFrame([
                    {'Feature': feature, 'Importance': importance}
                    for feature, importance in best_model_info['feature_importance'].items()
                ])
                importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
                return importance_df
        
        # Aggregate feature importance from all models
        all_importance = {}
        for model_name, model_info in self.trained_models.items():
            if model_info['feature_importance']:
                for feature, importance in model_info['feature_importance'].items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(importance)
        
        # Calculate average importance
        avg_importance = {
            feature: np.mean(importance_list) 
            for feature, importance_list in all_importance.items()
        }
        
        if avg_importance:
            importance_df = pd.DataFrame([
                {'Feature': feature, 'Importance': importance}
                for feature, importance in avg_importance.items()
            ])
            importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
            return importance_df
        
        return pd.DataFrame()
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make predictions using specified model or best model"""
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No trained models available")
            model = self.best_model['model']
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not found in trained models")
            model = self.trained_models[model_name]['model']
        
        return model.predict(X)
    
    def register_best_model_to_mlflow(self, model_name: str = "dynamic_pricing_model", 
                                     stage: str = "Staging") -> bool:
        """Register the best model to MLflow model registry"""
        if not self.mlflow_enabled or not self.best_model:
            logger.warning("MLflow not enabled or no best model available")
            return False
            
        try:
            with mlflow.start_run(run_name=f"best_model_registration_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                best_model_obj = self.best_model['model']
                
                # Log the best model
                if 'xgb' in self.best_model['name'].lower() and XGBOOST_AVAILABLE:
                    model_info = mlflow.xgboost.log_model(
                        best_model_obj, 
                        "model",
                        registered_model_name=model_name
                    )
                elif 'lightgbm' in self.best_model['name'].lower() and LIGHTGBM_AVAILABLE:
                    model_info = mlflow.lightgbm.log_model(
                        best_model_obj,
                        "model", 
                        registered_model_name=model_name
                    )
                else:
                    model_info = mlflow.sklearn.log_model(
                        best_model_obj,
                        "model",
                        registered_model_name=model_name
                    )
                
                # Log metadata
                mlflow.log_params(self.best_model['params'])
                mlflow.log_metric("cv_r2_score", self.best_model['cv_score'])
                
                # Set tags
                mlflow.set_tag("model_type", self.best_model['name'])
                mlflow.set_tag("stage", "production_candidate")
                mlflow.set_tag("registered", "true")
                
                logger.info(f"Best model '{self.best_model['name']}' registered as '{model_name}' in MLflow")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register model to MLflow: {e}")
            return False
            
    def get_mlflow_info(self) -> Dict[str, Any]:
        """Get MLflow tracking information"""
        if not self.mlflow_enabled:
            return {"mlflow_enabled": False}
            
        try:
            return {
                "mlflow_enabled": True,
                "experiment_name": self.experiment_name,
                "tracking_uri": mlflow.get_tracking_uri(),
                "experiment_id": mlflow.get_experiment_by_name(self.experiment_name).experiment_id if mlflow.get_experiment_by_name(self.experiment_name) else None
            }
        except Exception as e:
            return {
                "mlflow_enabled": True,
                "error": str(e)
            }
            
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        summary = {
            'total_models_trained': len(self.trained_models),
            'best_model': self.best_model,
            'model_performance': self.model_performance,
            'available_models': list(self.trained_models.keys())
        }
        
        # Add MLflow information
        if self.mlflow_enabled:
            summary['mlflow_info'] = self.get_mlflow_info()
            
        return summary
