"""
Unified Dynamic Pricing Pipeline
Main pipeline class that combines all modules into a single cohesive system
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Union, Tuple, Optional
from pathlib import Path
import warnings

from .data_processor import UnifiedDataProcessor
from .feature_engineer import UnifiedFeatureEngineer
from .model_trainer import UnifiedModelTrainer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class UnifiedDynamicPricingPipeline:
    """Complete unified dynamic pricing pipeline combining all modules"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize pipeline components
        self.data_processor = UnifiedDataProcessor(config.get('data_processor', {}))
        self.feature_engineer = UnifiedFeatureEngineer(config.get('feature_engineer', {}))
        self.model_trainer = UnifiedModelTrainer(config.get('model_trainer', {}))
        
        # Pipeline state
        self.pipeline_results = {}
        self.is_fitted = False
        
    def load_data(self, data_source: Union[str, pd.DataFrame, Dict[str, str]]) -> pd.DataFrame:
        """Load data from various sources"""
        logger.info("Loading data into pipeline")
        
        if isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        elif isinstance(data_source, str):
            # Load from file path
            if data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
                df = pd.read_excel(data_source)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        elif isinstance(data_source, dict):
            # Load multiple datasets and combine
            dataframes = []
            for name, path in data_source.items():
                if path.endswith('.csv'):
                    temp_df = pd.read_csv(path)
                elif path.endswith('.xlsx') or path.endswith('.xls'):
                    temp_df = pd.read_excel(path)
                else:
                    continue
                temp_df['data_source'] = name
                dataframes.append(temp_df)
            
            if dataframes:
                df = pd.concat(dataframes, ignore_index=True)
            else:
                raise ValueError("No valid data files found in dictionary")
        else:
            raise ValueError("data_source must be DataFrame, file path, or dictionary of file paths")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and schema"""
        logger.info("Validating data quality and schema")
        
        validation_results = self.data_processor.validate_data_schema(df)
        
        # Additional business logic validation for pricing data
        validation_results['business_validation'] = self._validate_business_rules(df)
        validation_results['overall_valid'] = (
            validation_results['schema_valid'] and 
            validation_results['quality_score'] > 50 and
            validation_results['business_validation']['valid']
        )
        
        return validation_results
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate business-specific rules for pricing data"""
        business_validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check for negative prices
        price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'selling'])]
        for price_col in price_cols:
            if price_col in df.columns:
                negative_prices = (df[price_col] < 0).sum()
                if negative_prices > 0:
                    business_validation['errors'].append(f"Found {negative_prices} negative prices in {price_col}")
                    business_validation['valid'] = False
        
        # Check for unrealistic price ranges
        for price_col in price_cols:
            if price_col in df.columns:
                max_price = df[price_col].max()
                if max_price > 100000:  # Adjust threshold as needed
                    business_validation['warnings'].append(f"Very high prices detected in {price_col}: max=${max_price:,.2f}")
        
        # Check for missing critical data
        demand_cols = [col for col in df.columns if any(x in col.lower() for x in ['demand', 'units', 'sold'])]
        if not demand_cols:
            business_validation['warnings'].append("No demand/sales data detected")
        
        return business_validation
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data using unified data processor"""
        logger.info("Preprocessing data")
        
        df_processed, processing_report = self.data_processor.fit_transform(df)
        
        # Store processing report
        self.pipeline_results['data_processing'] = processing_report
        
        return df_processed
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features using unified feature engineer"""
        logger.info("Engineering features for pricing optimization")
        
        df_features = self.feature_engineer.fit_transform(df)
        
        # Store feature engineering summary
        self.pipeline_results['feature_engineering'] = self.feature_engineer.get_feature_summary()
        
        return df_features
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = None, 
                            test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for model training"""
        logger.info("Preparing training and test datasets")
        
        # Auto-detect target column if not specified
        if target_column is None:
            target_candidates = [col for col in df.columns if any(x in col.lower() for x in ['price', 'selling'])]
            if target_candidates:
                target_column = target_candidates[0]
            else:
                raise ValueError("Target column not specified and could not be auto-detected")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Remove non-predictive columns and ensure only numeric features
        exclude_cols = [target_column, 'data_source'] + [col for col in df.columns if 'date' in col.lower()]
        
        # Get all potential feature columns
        potential_feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Only keep numeric columns and properly encoded categorical columns
        numeric_columns = df[potential_feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        # Add encoded categorical columns (those ending with '_encoded')
        encoded_categorical_columns = [col for col in potential_feature_columns if col.endswith('_encoded')]
        
        # Combine numeric and encoded categorical columns
        feature_columns = list(set(numeric_columns + encoded_categorical_columns))
        
        # Log what features we're using
        logger.info(f"Selected {len(feature_columns)} numeric/encoded features for training")
        logger.info(f"Excluded columns: {[col for col in potential_feature_columns if col not in feature_columns][:10]}...")  # Show first 10
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle any remaining missing values in features
        X = X.fillna(X.median(numeric_only=True))
        
        # Final check: ensure all columns are numeric
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            logger.warning(f"Removing non-numeric columns that were not properly encoded: {non_numeric_cols}")
            X = X.select_dtypes(include=[np.number])
        
        # Split data (use simple train_test_split for now, can be enhanced for time series)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Training data prepared. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Final feature columns: {list(X_train.columns)[:10]}...")  # Show first 10 features
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train all models using unified model trainer"""
        logger.info("Training multiple ML models")
        
        training_results = self.model_trainer.train_all_models(X_train, y_train)
        
        # Store training results
        self.pipeline_results['model_training'] = {
            'training_results': training_results,
            'best_model': self.model_trainer.best_model,
            'model_summary': self.model_trainer.get_training_summary()
        }
        
        return training_results
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate all trained models"""
        logger.info("Evaluating trained models")
        
        evaluation_results = self.model_trainer.evaluate_models(X_test, y_test)
        
        # Store evaluation results
        self.pipeline_results['model_evaluation'] = evaluation_results
        
        return evaluation_results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from trained models"""
        return self.model_trainer.get_feature_importance(top_n)
    
    def run_complete_pipeline(self, data_source: Union[str, pd.DataFrame, Dict[str, str]], 
                            target_column: str = None, 
                            test_size: float = 0.2) -> Dict[str, Any]:
        """Run the complete unified dynamic pricing pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING UNIFIED DYNAMIC PRICING PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load data
            df = self.load_data(data_source)
            
            # Step 2: Validate data
            validation_results = self.validate_data(df)
            
            if not validation_results['overall_valid']:
                logger.warning("Data validation issues detected, but continuing with pipeline")
            
            # Step 3: Preprocess data
            df_processed = self.preprocess_data(df)
            
            # Step 4: Engineer features
            df_features = self.engineer_features(df_processed)
            
            # Step 5: Prepare training data
            X_train, X_test, y_train, y_test = self.prepare_training_data(
                df_features, target_column, test_size
            )
            
            # Step 6: Train models
            training_results = self.train_models(X_train, y_train)
            
            # Step 7: Evaluate models
            evaluation_results = self.evaluate_models(X_test, y_test)
            
            # Step 8: Get feature importance
            feature_importance = self.get_feature_importance()
            
            # Step 9: Model comparison
            model_comparison = self.model_trainer.get_model_comparison()
            
            # Step 10: Register best model to MLflow (if enabled)
            model_registered = False
            if self.model_trainer.mlflow_enabled and self.model_trainer.best_model:
                model_registered = self.model_trainer.register_best_model_to_mlflow()
            
            # Mark as fitted
            self.is_fitted = True
            
            # Compile final results
            pipeline_results = {
                'pipeline_status': 'success',
                'data_validation': validation_results,
                'data_shapes': {
                    'original': df.shape,
                    'processed': df_processed.shape,
                    'features': df_features.shape,
                    'training': X_train.shape,
                    'testing': X_test.shape
                },
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'best_model': self.model_trainer.best_model,
                'feature_importance': feature_importance.to_dict('records') if not feature_importance.empty else [],
                'model_comparison': model_comparison.to_dict('records') if not model_comparison.empty else [],
                'feature_engineering_summary': self.feature_engineer.get_feature_summary(),
                'processing_summary': self.pipeline_results.get('data_processing', {}),
                'mlflow_info': self.model_trainer.get_mlflow_info(),
                'model_registered_to_mlflow': model_registered
            }
            
            self.pipeline_results.update(pipeline_results)
            
            logger.info("=" * 60)
            logger.info("UNIFIED DYNAMIC PRICING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Best model: {self.model_trainer.best_model['name'] if self.model_trainer.best_model else 'None'}")
            logger.info("=" * 60)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'pipeline_status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def predict(self, new_data: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make predictions on new data using trained models"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before making predictions")
        
        # Preprocess new data
        df_processed = self.data_processor.transform(new_data)
        df_features = self.feature_engineer.fit_transform(df_processed)
        
        # Remove target and non-predictive columns, keeping only numeric features
        exclude_cols = ['data_source'] + [col for col in df_features.columns if 'date' in col.lower()]
        
        # Get all potential feature columns
        potential_feature_columns = [col for col in df_features.columns if col not in exclude_cols]
        
        # Only keep numeric columns and properly encoded categorical columns
        numeric_columns = df_features[potential_feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        # Add encoded categorical columns (those ending with '_encoded')
        encoded_categorical_columns = [col for col in potential_feature_columns if col.endswith('_encoded')]
        
        # Combine numeric and encoded categorical columns
        feature_columns = list(set(numeric_columns + encoded_categorical_columns))
        
        X_new = df_features[feature_columns].copy()
        
        # Handle missing values
        X_new = X_new.fillna(X_new.median(numeric_only=True))
        
        # Final check: ensure all columns are numeric
        non_numeric_cols = X_new.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            logger.warning(f"Removing non-numeric columns from prediction data: {non_numeric_cols}")
            X_new = X_new.select_dtypes(include=[np.number])
        
        # Make predictions
        predictions = self.model_trainer.predict(X_new, model_name)
        
        return predictions
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary"""
        if not self.pipeline_results:
            return {"message": "Pipeline has not been run yet"}
        
        summary = {
            'pipeline_fitted': self.is_fitted,
            'data_quality_score': self.pipeline_results.get('data_validation', {}).get('quality_score', 0),
            'total_features_created': len(self.feature_engineer.created_features),
            'models_trained': len(self.model_trainer.trained_models),
            'best_model': self.model_trainer.best_model['name'] if self.model_trainer.best_model else None,
            'best_model_score': self.model_trainer.best_model['cv_score'] if self.model_trainer.best_model else None,
            'pipeline_components': {
                'data_processor': type(self.data_processor).__name__,
                'feature_engineer': type(self.feature_engineer).__name__,
                'model_trainer': type(self.model_trainer).__name__
            }
        }
        
        return summary
    
    def save_pipeline(self, filepath: str):
        """Save the fitted pipeline to disk"""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        import joblib
        
        pipeline_state = {
            'config': self.config,
            'data_processor': self.data_processor,
            'feature_engineer': self.feature_engineer,
            'model_trainer': self.model_trainer,
            'pipeline_results': self.pipeline_results,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(pipeline_state, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str):
        """Load a fitted pipeline from disk"""
        import joblib
        
        pipeline_state = joblib.load(filepath)
        
        # Create new instance
        pipeline = cls(pipeline_state['config'])
        
        # Restore state
        pipeline.data_processor = pipeline_state['data_processor']
        pipeline.feature_engineer = pipeline_state['feature_engineer']
        pipeline.model_trainer = pipeline_state['model_trainer']
        pipeline.pipeline_results = pipeline_state['pipeline_results']
        pipeline.is_fitted = pipeline_state['is_fitted']
        
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline
