"""
Azure-Enhanced Dynamic Pricing Pipeline
Main pipeline with full Azure services integration
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Union, Tuple, Optional
from pathlib import Path
import warnings
import time
from contextlib import nullcontext

# Import Azure integrations
from ..azure_integrations import (
    ADLSManager, AzureKeyVaultManager, AzureMLManager, 
    DatabricksManager, AzureMonitoringManager, MonitoredOperation
)
from ..config import get_azure_config, validate_azure_setup

# Import original pipeline components
from .dynamic_pricing_pipeline import UnifiedDynamicPricingPipeline

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AzureEnhancedPricingPipeline(UnifiedDynamicPricingPipeline):
    """Azure-enhanced dynamic pricing pipeline with full cloud integration"""
    
    def __init__(self, config: Dict[str, Any] = None, azure_config: Dict[str, Any] = None):
        
        # Get Azure configuration
        self.azure_config_manager = get_azure_config()
        self.azure_features = self.azure_config_manager.get_features()
        
        # Validate Azure setup
        self.azure_status = validate_azure_setup()
        
        # Initialize Azure services
        self.adls_manager = None
        self.keyvault_manager = None
        self.aml_manager = None
        self.databricks_manager = None
        self.monitoring_manager = None
        
        self._initialize_azure_services()
        
        # Initialize base pipeline with Azure-enhanced config
        enhanced_config = self._get_enhanced_config(config)
        super().__init__(enhanced_config)
        
        logger.info("Azure-enhanced pricing pipeline initialized")
        
    def _initialize_azure_services(self):
        """Initialize Azure services based on configuration"""
        
        # Initialize ADLS
        if self.azure_features.get('enable_adls', False):
            try:
                adls_config = self.azure_config_manager.get_adls_config()
                self.adls_manager = ADLSManager(
                    storage_account_name=adls_config.get('storage_account_name'),
                    container_name=adls_config.get('container_name', 'pricing-data')
                )
                logger.info("ADLS manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ADLS: {e}")
        
        # Initialize Key Vault
        if self.azure_features.get('enable_keyvault', False):
            try:
                kv_config = self.azure_config_manager.get_keyvault_config()
                if kv_config.get('vault_url'):
                    self.keyvault_manager = AzureKeyVaultManager(
                        vault_url=kv_config['vault_url']
                    )
                    logger.info("Key Vault manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Key Vault: {e}")
        
        # Initialize Azure ML
        if self.azure_features.get('enable_aml', False):
            try:
                aml_config = self.azure_config_manager.get_aml_config()
                self.aml_manager = AzureMLManager(
                    subscription_id=aml_config.get('subscription_id'),
                    resource_group=aml_config.get('resource_group'),
                    workspace_name=aml_config.get('workspace_name')
                )
                logger.info("Azure ML manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Azure ML: {e}")
        
        # Initialize Databricks
        if self.azure_features.get('enable_databricks', False):
            try:
                db_config = self.azure_config_manager.get_databricks_config()
                self.databricks_manager = DatabricksManager(
                    server_hostname=db_config.get('server_hostname'),
                    http_path=db_config.get('http_path'),
                    access_token=db_config.get('access_token'),
                    workspace_url=db_config.get('workspace_url')
                )
                logger.info("Databricks manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Databricks: {e}")
        
        # Initialize Monitoring
        if self.azure_features.get('enable_monitoring', False):
            try:
                mon_config = self.azure_config_manager.get_monitoring_config()
                self.monitoring_manager = AzureMonitoringManager(
                    connection_string=mon_config.get('connection_string'),
                    instrumentation_key=mon_config.get('instrumentation_key'),
                    service_name=mon_config.get('service_name', 'dynamic-pricing-pipeline')
                )
                logger.info("Azure Monitor manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Azure Monitor: {e}")
    
    def _get_enhanced_config(self, base_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get configuration enhanced with Azure settings"""
        
        # Start with Azure pipeline configuration
        enhanced_config = self.azure_config_manager.get_pipeline_config()
        
        # Merge with base config if provided
        if base_config:
            # Deep merge configurations
            for section, section_config in base_config.items():
                if section in enhanced_config and isinstance(section_config, dict):
                    enhanced_config[section].update(section_config)
                else:
                    enhanced_config[section] = section_config
        
        return enhanced_config
    
    def load_data_from_azure(self, data_source: Union[str, pd.DataFrame, Dict[str, str]]) -> pd.DataFrame:
        """Load data with Azure services integration"""
        
        with MonitoredOperation(self.monitoring_manager, 'data_loading') if self.monitoring_manager else nullcontext():
            
            # If ADLS is enabled and data source is a path, try loading from ADLS
            if self.adls_manager and isinstance(data_source, str):
                if not data_source.startswith(('http', 'https', '/', 'C:', 'D:')):  # Assume ADLS path
                    logger.info(f"Attempting to load data from ADLS: {data_source}")
                    df = self.adls_manager.download_dataframe(data_source, format='parquet')
                    if df is not None:
                        logger.info(f"Successfully loaded data from ADLS with {len(df)} rows")
                        return df
                    else:
                        logger.warning(f"Failed to load from ADLS, falling back to local: {data_source}")
            
            # Fall back to original data loading
            return super().load_data(data_source)
    
    def get_azure_status(self) -> Dict[str, Any]:
        """Get comprehensive Azure integration status"""
        return {
            'azure_config_status': self.azure_status,
            'active_services': self._get_active_azure_services(),
            'service_health': {
                'adls': self.adls_manager is not None,
                'keyvault': self.keyvault_manager is not None,
                'aml': self.aml_manager is not None,
                'databricks': self.databricks_manager is not None,
                'monitoring': self.monitoring_manager is not None
            }
        }
    
    def _get_active_azure_services(self) -> List[str]:
        """Get list of active Azure services"""
        active_services = []
        
        if self.adls_manager:
            active_services.append('ADLS')
        if self.keyvault_manager:
            active_services.append('KeyVault')
        if self.aml_manager:
            active_services.append('AzureML')
        if self.databricks_manager:
            active_services.append('Databricks')
        if self.monitoring_manager:
            active_services.append('Monitor')
        
        return active_services
