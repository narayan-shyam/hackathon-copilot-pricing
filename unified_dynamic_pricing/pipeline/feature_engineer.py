"""
Unified Feature Engineering Pipeline
Advanced feature engineering for pricing optimization combining all modules
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class UnifiedFeatureEngineer:
    """Comprehensive feature engineering for pricing optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.created_features = []
        self.feature_importance = {}
        
    def create_pricing_elasticity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive pricing elasticity and demand sensitivity features"""
        df_features = df.copy()
        
        # Auto-detect price and demand columns
        price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'selling', 'mrp'])]
        demand_cols = [col for col in df.columns if any(x in col.lower() for x in ['demand', 'units', 'sold', 'quantity'])]
        cost_cols = [col for col in df.columns if 'cost' in col.lower()]
        competitor_cols = [col for col in df.columns if any(x in col.lower() for x in ['competitor', 'base', 'benchmark'])]
        
        if price_cols and demand_cols:
            price_col = price_cols[0]
            demand_col = demand_cols[0]
            
            # Price elasticity features
            df_features['PriceChange'] = df_features[price_col].pct_change()
            df_features['DemandChange'] = df_features[demand_col].pct_change()
            
            # Rolling statistics for trend analysis
            for window in [7, 14, 30]:
                df_features[f'Price_MA_{window}'] = df_features[price_col].rolling(window=window).mean()
                df_features[f'Demand_MA_{window}'] = df_features[demand_col].rolling(window=window).mean()
                df_features[f'Price_Volatility_{window}'] = df_features[price_col].rolling(window=window).std()
                df_features[f'Demand_Volatility_{window}'] = df_features[demand_col].rolling(window=window).std()
            
            # Price elasticity calculation (simplified)
            df_features['PriceElasticity'] = (df_features['DemandChange'] / 
                                           df_features['PriceChange'].replace(0, np.nan))
            df_features['PriceElasticity'] = df_features['PriceElasticity'].replace([np.inf, -np.inf], np.nan)
            
            # Price positioning features
            df_features['PricePercentile'] = df_features[price_col].rank(pct=True)
            df_features['DemandPercentile'] = df_features[demand_col].rank(pct=True)
            df_features['PriceDemandRatio'] = df_features[price_col] / (df_features[demand_col] + 1)
            
            # Revenue calculation
            df_features['Revenue'] = df_features[price_col] * df_features[demand_col]
            df_features['Revenue_MA_7'] = df_features['Revenue'].rolling(window=7).mean()
            df_features['RevenueTrend'] = (df_features['Revenue'].rolling(window=7).mean() / 
                                         df_features['Revenue'].rolling(window=30).mean())
            
            self.created_features.extend([
                'PriceChange', 'DemandChange', 'PriceElasticity', 'PricePercentile', 
                'DemandPercentile', 'PriceDemandRatio', 'Revenue', 'RevenueTrend'
            ])
        
        # Competitor analysis
        if competitor_cols and price_cols:
            competitor_col = competitor_cols[0]
            price_col = price_cols[0]
            
            df_features['PricePremium'] = df_features[price_col] - df_features[competitor_col]
            df_features['PricePremiumPct'] = ((df_features[price_col] - df_features[competitor_col]) / 
                                            (df_features[competitor_col] + 1e-8)) * 100
            df_features['CompetitivePosition'] = np.where(df_features[price_col] > df_features[competitor_col], 1, 0)
            
            self.created_features.extend(['PricePremium', 'PricePremiumPct', 'CompetitivePosition'])
        
        # Cost-based features
        if cost_cols and price_cols:
            cost_col = cost_cols[0]
            price_col = price_cols[0]
            
            df_features['ProfitMargin'] = df_features[price_col] - df_features[cost_col]
            df_features['ProfitMarginPct'] = ((df_features[price_col] - df_features[cost_col]) / 
                                            (df_features[price_col] + 1e-8)) * 100
            df_features['MarkupRatio'] = df_features[price_col] / (df_features[cost_col] + 1e-8)
            
            self.created_features.extend(['ProfitMargin', 'ProfitMarginPct', 'MarkupRatio'])
        
        logger.info("Pricing elasticity features created")
        return df_features
    
    def create_customer_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer behavior and engagement features"""
        df_features = df.copy()
        
        # Customer behavior metrics
        behavior_metrics = ['CTR', 'AbandonedCartRate', 'BounceRate', 'AvgSessionDuration', 'ConversionRate']
        available_metrics = [col for col in behavior_metrics if col in df.columns]
        
        if available_metrics:
            # Customer engagement score
            engagement_features = []
            
            if 'CTR' in df.columns:
                df_features['CTR_Norm'] = (df_features['CTR'] - df_features['CTR'].mean()) / (df_features['CTR'].std() + 1e-8)
                engagement_features.append('CTR_Norm')
            
            if 'BounceRate' in df.columns:
                df_features['BounceRate_Inv'] = 1 - df_features['BounceRate']
                engagement_features.append('BounceRate_Inv')
            
            if 'AbandonedCartRate' in df.columns:
                df_features['CartCompletion'] = 1 - df_features['AbandonedCartRate']
                engagement_features.append('CartCompletion')
            
            if any('session' in col.lower() for col in df.columns):
                session_cols = [col for col in df.columns if 'session' in col.lower()]
                session_col = session_cols[0]
                if 'sec' in session_col.lower():
                    df_features['SessionDuration_Min'] = df_features[session_col] / 60
                else:
                    df_features['SessionDuration_Min'] = df_features[session_col]
                engagement_features.append('SessionDuration_Min')
            
            # Composite customer engagement score
            if engagement_features:
                for feature in engagement_features:
                    col_min = df_features[feature].min()
                    col_max = df_features[feature].max()
                    if col_max != col_min:
                        df_features[f'{feature}_Scaled'] = ((df_features[feature] - col_min) / (col_max - col_min))
                    else:
                        df_features[f'{feature}_Scaled'] = 0
                
                scaled_features = [f'{f}_Scaled' for f in engagement_features]
                df_features['CustomerEngagementScore'] = df_features[scaled_features].mean(axis=1)
                
                self.created_features.extend(['CustomerEngagementScore'])
        
        # Customer lifetime value approximation
        price_cols = [col for col in df.columns if any(x in col.lower() for x in ['price', 'selling'])]
        demand_cols = [col for col in df.columns if any(x in col.lower() for x in ['demand', 'units', 'sold'])]
        
        if price_cols and demand_cols:
            price_col = price_cols[0]
            demand_col = demand_cols[0]
            
            if 'Revenue' not in df_features.columns:
                df_features['Revenue'] = df_features[price_col] * df_features[demand_col]
            
            # Customer value metrics
            window = self.config.get('ltv_window', 30)
            df_features['CustomerLTV'] = df_features['Revenue'].rolling(window=window).sum()
            df_features['AvgOrderValue'] = df_features['Revenue'] / (df_features[demand_col] + 1)
            df_features['PurchaseFrequency'] = df_features[demand_col].rolling(window=window).count()
            
            self.created_features.extend(['CustomerLTV', 'AvgOrderValue', 'PurchaseFrequency'])
        
        logger.info("Customer behavior features created")
        return df_features
    
    def create_inventory_optimization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create inventory management and optimization features"""
        df_features = df.copy()
        
        # Stock level features
        stock_cols = [col for col in df.columns if 'stock' in col.lower()]
        if len(stock_cols) >= 2:
            # Assuming we have start and end stock columns
            stock_start = stock_cols[0]
            stock_end = stock_cols[1] if len(stock_cols) > 1 else stock_cols[0]
            
            df_features['StockMovement'] = df_features[stock_start] - df_features[stock_end]
            df_features['StockTurnover'] = df_features['StockMovement'] / (df_features[stock_start] + 1)
            
            self.created_features.extend(['StockMovement', 'StockTurnover'])
        
        # Service level and demand fulfillment
        demand_cols = [col for col in df.columns if any(x in col.lower() for x in ['demand', 'units', 'sold'])]
        fulfilled_cols = [col for col in df.columns if 'fulfilled' in col.lower()]
        
        if demand_cols and fulfilled_cols:
            demand_col = demand_cols[0]
            fulfilled_col = fulfilled_cols[0]
            
            df_features['ServiceLevel'] = df_features[fulfilled_col] / (df_features[demand_col] + 1)
            df_features['StockoutRisk'] = np.where(df_features['ServiceLevel'] < 0.95, 1, 0)
            
            self.created_features.extend(['ServiceLevel', 'StockoutRisk'])
        
        # Demand forecasting features
        if demand_cols:
            demand_col = demand_cols[0]
            
            for window in [7, 14, 30]:
                df_features[f'Demand_MA_{window}'] = df_features[demand_col].rolling(window=window).mean()
                df_features[f'Demand_Std_{window}'] = df_features[demand_col].rolling(window=window).std()
                df_features[f'Demand_CV_{window}'] = (df_features[f'Demand_Std_{window}'] / 
                                                    (df_features[f'Demand_MA_{window}'] + 1))
            
            df_features['DemandTrend'] = (df_features['Demand_MA_7'] / 
                                        (df_features['Demand_MA_30'] + 1))
            df_features['DemandVolatility'] = df_features[demand_col].rolling(window=14).std()
            
            self.created_features.extend(['DemandTrend', 'DemandVolatility'])
        
        # Backorder features
        backorder_cols = [col for col in df.columns if 'backorder' in col.lower()]
        if backorder_cols and demand_cols:
            backorder_col = backorder_cols[0]
            demand_col = demand_cols[0]
            
            df_features['BackorderRate'] = df_features[backorder_col] / (df_features[demand_col] + 1)
            self.created_features.extend(['BackorderRate'])
        
        logger.info("Inventory optimization features created")
        return df_features
    
    def create_time_series_features(self, df: pd.DataFrame, target_cols: List[str] = None, 
                                  lag_periods: List[int] = None) -> pd.DataFrame:
        """Create comprehensive time series features including lags and rolling statistics"""
        df_features = df.copy()
        
        # Auto-detect target columns if not specified
        if target_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_cols = [col for col in numeric_cols 
                          if any(x in col.lower() for x in ['price', 'demand', 'units', 'sold', 'revenue'])]
        
        lag_periods = lag_periods or [1, 3, 7, 14]
        
        for col in target_cols:
            if col in df.columns:
                # Lag features
                for lag in lag_periods:
                    df_features[f'{col}_Lag_{lag}'] = df_features[col].shift(lag)
                
                # Rolling statistics
                for window in [3, 7, 14, 30]:
                    df_features[f'{col}_MA_{window}'] = df_features[col].rolling(window=window).mean()
                    df_features[f'{col}_Std_{window}'] = df_features[col].rolling(window=window).std()
                    
                # Rolling lag features (lagged rolling means)
                for window in [7, 14]:
                    df_features[f'{col}_LagMA_{window}'] = df_features[col].shift(1).rolling(window=window).mean()
                
                # Trend and momentum features
                df_features[f'{col}_Momentum_3'] = df_features[col] - df_features[col].shift(3)
                df_features[f'{col}_Momentum_7'] = df_features[col] - df_features[col].shift(7)
                
                # Rate of change
                df_features[f'{col}_ROC_3'] = df_features[col].pct_change(periods=3)
                df_features[f'{col}_ROC_7'] = df_features[col].pct_change(periods=7)
        
        logger.info(f"Time series features created for {len(target_cols)} columns")
        return df_features
    
    def create_seasonal_features(self, df: pd.DataFrame, date_column: str = None) -> pd.DataFrame:
        """Create seasonal and calendar-based features"""
        df_features = df.copy()
        
        # Auto-detect date column if not specified
        if date_column is None:
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                date_column = date_columns[0]
        
        if date_column and date_column in df_features.columns:
            df_features[date_column] = pd.to_datetime(df_features[date_column])
            
            # Seasonal indicators
            df_features['IsPeakSeason'] = df_features[date_column].dt.month.isin([11, 12, 1]).astype(int)
            df_features['IsSummer'] = df_features[date_column].dt.month.isin([6, 7, 8]).astype(int)
            df_features['IsHolidaySeason'] = df_features[date_column].dt.month.isin([11, 12]).astype(int)
            
            # Holiday proximity (simplified)
            df_features['DaysToYearEnd'] = (df_features[date_column] - 
                                          pd.to_datetime(df_features[date_column].dt.year.astype(str) + '-12-31')).dt.days
            df_features['DaysFromYearStart'] = (df_features[date_column] - 
                                              pd.to_datetime(df_features[date_column].dt.year.astype(str) + '-01-01')).dt.days
            
            self.created_features.extend(['IsPeakSeason', 'IsSummer', 'IsHolidaySeason', 
                                        'DaysToYearEnd', 'DaysFromYearStart'])
        
        logger.info("Seasonal features created")
        return df_features
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete unified feature engineering pipeline"""
        logger.info("Starting unified feature engineering pipeline")
        
        # Apply all feature engineering steps
        df_features = self.create_pricing_elasticity_features(df)
        df_features = self.create_customer_behavior_features(df_features)
        df_features = self.create_inventory_optimization_features(df_features)
        df_features = self.create_time_series_features(df_features)
        df_features = self.create_seasonal_features(df_features)
        
        # Remove duplicate features that might have been created
        df_features = df_features.loc[:, ~df_features.columns.duplicated()]
        
        # Handle infinite values that might have been created during feature engineering
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        df_features[numeric_cols] = df_features[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values in newly created features with appropriate defaults
        for feature in self.created_features:
            if feature in df_features.columns:
                if df_features[feature].isna().any():
                    # Use median for most features, 0 for ratios and percentages
                    if any(x in feature.lower() for x in ['rate', 'ratio', 'pct', 'premium']):
                        df_features[feature] = df_features[feature].fillna(0)
                    else:
                        df_features[feature] = df_features[feature].fillna(df_features[feature].median())
        
        logger.info(f"Feature engineering completed. Created {len(self.created_features)} new features")
        logger.info(f"Total features: {len(df_features.columns)}")
        
        return df_features
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of created features"""
        return {
            'total_created_features': len(self.created_features),
            'created_features': self.created_features,
            'feature_categories': {
                'pricing_elasticity': len([f for f in self.created_features if any(x in f.lower() for x in ['price', 'elasticity', 'revenue'])]),
                'customer_behavior': len([f for f in self.created_features if any(x in f.lower() for x in ['customer', 'engagement', 'ltv'])]),
                'inventory': len([f for f in self.created_features if any(x in f.lower() for x in ['stock', 'demand', 'service'])]),
                'seasonal': len([f for f in self.created_features if any(x in f.lower() for x in ['season', 'holiday', 'days'])])
            }
        }
