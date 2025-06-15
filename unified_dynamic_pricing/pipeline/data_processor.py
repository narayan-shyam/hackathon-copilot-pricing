"""
Unified Data Processing Pipeline
Consolidated data validation, preprocessing and quality assessment
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class UnifiedDataProcessor:
    """Comprehensive data processing pipeline combining validation, preprocessing and quality assessment"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.validation_rules = {
            'required_columns': ['price', 'demand', 'cost', 'competitor_price', 'date'],
            'numeric_columns': ['price', 'demand', 'cost', 'competitor_price'],
            'date_columns': ['date'],
            'price_range': (0, 10000),
            'demand_range': (0, float('inf')),
            'cost_range': (0, float('inf'))
        }
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.preprocessing_stats = {}
        self.quality_report = {}
    
    def validate_data_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data schema and quality validation"""
        validation_results = {
            'schema_valid': True,
            'missing_columns': [],
            'incorrect_types': [],
            'quality_score': 0,
            'recommendations': []
        }
        
        # Check required columns (flexible approach)
        available_cols = set(df.columns)
        price_cols = [col for col in available_cols if any(x in col.lower() for x in ['price', 'selling', 'mrp'])]
        demand_cols = [col for col in available_cols if any(x in col.lower() for x in ['demand', 'units', 'sold', 'quantity'])]
        
        if not price_cols:
            validation_results['missing_columns'].append('price_related_column')
            validation_results['schema_valid'] = False
        
        if not demand_cols:
            validation_results['missing_columns'].append('demand_related_column')
            validation_results['schema_valid'] = False
        
        # Data quality assessment
        quality_metrics = self._assess_data_quality(df)
        validation_results.update(quality_metrics)
        
        logger.info(f"Schema validation: {'PASSED' if validation_results['schema_valid'] else 'FAILED'}")
        logger.info(f"Data quality score: {validation_results['quality_score']:.2f}/100")
        
        return validation_results
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        quality_metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'outliers_detected': {},
            'quality_score': 0
        }
        
        # Outlier detection using IQR method
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            quality_metrics['outliers_detected'][col] = len(outliers)
        
        # Calculate overall quality score
        quality_score = 100
        quality_score -= min(quality_metrics['missing_data_percentage'] * 2, 50)
        quality_score -= min((quality_metrics['duplicate_rows'] / len(df)) * 100, 20)
        quality_metrics['quality_score'] = max(quality_score, 0)
        
        return quality_metrics
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """Intelligent missing value handling with multiple strategies"""
        df_processed = df.copy()
        missing_info = {}
        
        for column in df_processed.columns:
            missing_count = df_processed[column].isnull().sum()
            missing_percentage = (missing_count / len(df_processed)) * 100
            missing_info[column] = {'count': missing_count, 'percentage': missing_percentage}
            
            if missing_count > 0:
                if strategy == 'auto':
                    if pd.api.types.is_numeric_dtype(df_processed[column]):
                        if missing_percentage < 5:
                            df_processed[column].fillna(df_processed[column].median(), inplace=True)
                        elif missing_percentage < 20:
                            imputer = KNNImputer(n_neighbors=5)
                            df_processed[column] = imputer.fit_transform(df_processed[[column]]).flatten()
                            self.imputers[column] = imputer
                        else:
                            df_processed[column].fillna(method='ffill', inplace=True)
                            df_processed[column].fillna(df_processed[column].median(), inplace=True)
                    else:
                        mode_val = df_processed[column].mode()
                        if len(mode_val) > 0:
                            df_processed[column].fillna(mode_val[0], inplace=True)
        
        self.preprocessing_stats['missing_values'] = missing_info
        logger.info(f"Missing values handled using {strategy} strategy")
        return df_processed
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Advanced outlier handling with business context"""
        df_processed = df.copy()
        outlier_info = {}
        
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in numeric_columns:
            if method == 'iqr':
                Q1 = df_processed[column].quantile(0.25)
                Q3 = df_processed[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing (better for pricing data)
                df_processed[column] = df_processed[column].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                mean_val = df_processed[column].mean()
                std_val = df_processed[column].std()
                threshold = mean_val + 3 * std_val
                df_processed[column] = df_processed[column].clip(upper=threshold)
            
            outlier_info[column] = {
                'method': method,
                'original_range': (df[column].min(), df[column].max()),
                'processed_range': (df_processed[column].min(), df_processed[column].max())
            }
        
        self.preprocessing_stats['outliers'] = outlier_info
        logger.info(f"Outliers handled using {method} method")
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart categorical encoding based on cardinality"""
        df_processed = df.copy()
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove date columns from categorical encoding
        categorical_columns = [col for col in categorical_columns if 'date' not in col.lower()]
        
        encoded_columns = []  # Track which columns we encode
        
        for column in categorical_columns:
            unique_values = df_processed[column].nunique()
            
            if unique_values <= 10:
                # Label Encoding for low cardinality
                encoder = LabelEncoder()
                df_processed[column + '_encoded'] = encoder.fit_transform(df_processed[column].astype(str))
                self.encoders[column] = encoder
                encoded_columns.append(column)
            elif unique_values <= 50:
                # One-Hot Encoding for medium cardinality
                dummies = pd.get_dummies(df_processed[column], prefix=column)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                self.encoders[column] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
                encoded_columns.append(column)
            else:
                # Label Encoding for high cardinality with warning
                logger.warning(f"High cardinality column {column} detected")
                encoder = LabelEncoder()
                df_processed[column + '_encoded'] = encoder.fit_transform(df_processed[column].astype(str))
                self.encoders[column] = encoder
                encoded_columns.append(column)
        
        # Remove original categorical columns that were encoded
        df_processed = df_processed.drop(columns=encoded_columns)
        
        logger.info(f"Categorical encoding completed for {len(categorical_columns)} columns")
        logger.info(f"Original categorical columns removed: {encoded_columns}")
        return df_processed
    
    def create_temporal_features(self, df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
        """Create comprehensive temporal features"""
        df_processed = df.copy()
        
        # Auto-detect date column if not specified
        if date_column is None:
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                date_column = date_columns[0]
        
        if date_column and date_column in df_processed.columns:
            df_processed[date_column] = pd.to_datetime(df_processed[date_column])
            
            # Extract temporal features
            df_processed['year'] = df_processed[date_column].dt.year
            df_processed['month'] = df_processed[date_column].dt.month
            df_processed['day'] = df_processed[date_column].dt.day
            df_processed['day_of_week'] = df_processed[date_column].dt.dayofweek
            df_processed['day_of_year'] = df_processed[date_column].dt.dayofyear
            df_processed['week_of_year'] = df_processed[date_column].dt.isocalendar().week
            df_processed['quarter'] = df_processed[date_column].dt.quarter
            
            # Cyclical features for better ML performance
            df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
            df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
            df_processed['dow_sin'] = np.sin(2 * np.pi * df_processed['day_of_week'] / 7)
            df_processed['dow_cos'] = np.cos(2 * np.pi * df_processed['day_of_week'] / 7)
            
            # Business temporal features
            df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
            df_processed['is_month_end'] = (df_processed[date_column].dt.day >= 28).astype(int)
            df_processed['is_quarter_end'] = df_processed[date_column].dt.month.isin([3, 6, 9, 12]).astype(int)
            
            logger.info("Temporal features created successfully")
        
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, method: str = 'robust') -> pd.DataFrame:
        """Feature scaling with multiple methods"""
        df_processed = df.copy()
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude encoded categorical columns from scaling
        numeric_columns = [col for col in numeric_columns if not col.endswith('_encoded')]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Scaling method must be 'standard' or 'robust'")
        
        if numeric_columns:
            df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
            self.scalers['features'] = scaler
            
        logger.info(f"Feature scaling completed using {method} method")
        return df_processed
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Complete unified data processing pipeline"""
        logger.info("Starting unified data processing pipeline")
        
        # Step 1: Validate data schema and quality
        validation_results = self.validate_data_schema(df)
        
        # Step 2: Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Step 3: Handle outliers
        df_processed = self.handle_outliers(df_processed)
        
        # Step 4: Create temporal features
        df_processed = self.create_temporal_features(df_processed)
        
        # Step 5: Encode categorical features
        df_processed = self.encode_categorical_features(df_processed)
        
        # Step 6: Scale numerical features
        df_processed = self.scale_features(df_processed)
        
        processing_report = {
            'validation_results': validation_results,
            'preprocessing_stats': self.preprocessing_stats,
            'input_shape': df.shape,
            'output_shape': df_processed.shape,
            'created_features': df_processed.columns.tolist()
        }
        
        logger.info(f"Data processing completed. Shape: {df.shape} -> {df_processed.shape}")
        return df_processed, processing_report
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted processors"""
        df_processed = df.copy()
        
        # Apply the same preprocessing steps with fitted processors
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.handle_outliers(df_processed)
        df_processed = self.create_temporal_features(df_processed)
        
        # Apply fitted encoders
        encoded_columns = []  # Track which columns we encode
        
        for column, encoder in self.encoders.items():
            if column in df_processed.columns:
                if isinstance(encoder, dict) and encoder.get('type') == 'onehot':
                    dummies = pd.get_dummies(df_processed[column], prefix=column)
                    for col in encoder['columns']:
                        if col not in dummies.columns:
                            dummies[col] = 0
                    df_processed = pd.concat([df_processed, dummies[encoder['columns']]], axis=1)
                else:
                    df_processed[column + '_encoded'] = encoder.transform(df_processed[column].astype(str))
                encoded_columns.append(column)
        
        # Remove original categorical columns that were encoded
        df_processed = df_processed.drop(columns=[col for col in encoded_columns if col in df_processed.columns])
        
        # Apply fitted scalers
        if 'features' in self.scalers:
            numeric_columns = [col for col in df_processed.select_dtypes(include=[np.number]).columns 
                             if not col.endswith('_encoded')]
            if numeric_columns:
                df_processed[numeric_columns] = self.scalers['features'].transform(df_processed[numeric_columns])
        
        return df_processed
