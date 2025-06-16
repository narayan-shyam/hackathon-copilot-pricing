"""
Azure Configuration for Dynamic Pricing Pipeline
Centralized configuration for all Azure services
"""

import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class AzureConfig:
    """Azure services configuration manager"""
    
    def __init__(self, use_env_variables: bool = True):
        self.use_env_variables = use_env_variables
        self._config = {}
        
        if use_env_variables:
            self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # Azure Data Lake Storage
        self._config['adls'] = {
            'storage_account_name': os.getenv('AZURE_STORAGE_ACCOUNT_NAME'),
            'storage_account_key': os.getenv('AZURE_STORAGE_ACCOUNT_KEY'),
            'container_name': os.getenv('AZURE_ADLS_CONTAINER', 'pricing-data'),
            'connection_string': os.getenv('AZURE_ADLS_CONNECTION_STRING')
        }
        
        # Azure Key Vault
        self._config['keyvault'] = {
            'vault_url': os.getenv('AZURE_KEYVAULT_URL'),
            'client_id': os.getenv('AZURE_CLIENT_ID'),
            'client_secret': os.getenv('AZURE_CLIENT_SECRET'),
            'tenant_id': os.getenv('AZURE_TENANT_ID')
        }
        
        # Azure Machine Learning
        self._config['aml'] = {
            'subscription_id': os.getenv('AZURE_ML_SUBSCRIPTION_ID'),
            'resource_group': os.getenv('AZURE_ML_RESOURCE_GROUP'),
            'workspace_name': os.getenv('AZURE_ML_WORKSPACE_NAME'),
            'compute_target': os.getenv('AZURE_ML_COMPUTE_TARGET', 'cpu-cluster')
        }
        
        # Azure Databricks
        self._config['databricks'] = {
            'workspace_url': os.getenv('DATABRICKS_WORKSPACE_URL'),
            'access_token': os.getenv('DATABRICKS_ACCESS_TOKEN'),
            'server_hostname': os.getenv('DATABRICKS_SERVER_HOSTNAME'),
            'http_path': os.getenv('DATABRICKS_HTTP_PATH'),
            'cluster_id': os.getenv('DATABRICKS_CLUSTER_ID')
        }
        
        # Azure Monitor
        self._config['monitoring'] = {
            'connection_string': os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING'),
            'instrumentation_key': os.getenv('APPINSIGHTS_INSTRUMENTATIONKEY'),
            'service_name': os.getenv('AZURE_MONITOR_SERVICE_NAME', 'dynamic-pricing-pipeline')
        }
        
        # Feature flags
        self._config['features'] = {
            'enable_adls': os.getenv('ENABLE_ADLS', 'true').lower() == 'true',
            'enable_aml': os.getenv('ENABLE_AML', 'true').lower() == 'true',
            'enable_databricks': os.getenv('ENABLE_DATABRICKS', 'false').lower() == 'true',
            'enable_monitoring': os.getenv('ENABLE_MONITORING', 'true').lower() == 'true',
            'enable_keyvault': os.getenv('ENABLE_KEYVAULT', 'true').lower() == 'true'
        }
    
    def get_adls_config(self) -> Dict[str, Any]:
        """Get ADLS configuration"""
        return self._config.get('adls', {})
    
    def get_keyvault_config(self) -> Dict[str, Any]:
        """Get Key Vault configuration"""
        return self._config.get('keyvault', {})
    
    def get_aml_config(self) -> Dict[str, Any]:
        """Get Azure ML configuration"""
        return self._config.get('aml', {})
    
    def get_databricks_config(self) -> Dict[str, Any]:
        """Get Databricks configuration"""
        return self._config.get('databricks', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self._config.get('monitoring', {})
    
    def get_features(self) -> Dict[str, bool]:
        """Get feature flags"""
        return self._config.get('features', {})
    
    def is_azure_enabled(self) -> bool:
        """Check if any Azure service is enabled"""
        features = self.get_features()
        return any([
            features.get('enable_adls', False),
            features.get('enable_aml', False),
            features.get('enable_databricks', False),
            features.get('enable_monitoring', False),
            features.get('enable_keyvault', False)
        ])
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate Azure configuration"""
        validation = {}
        features = self.get_features()
        
        # Validate ADLS
        if features.get('enable_adls'):
            adls_config = self.get_adls_config()
            validation['adls_valid'] = bool(
                adls_config.get('storage_account_name') and 
                (adls_config.get('storage_account_key') or adls_config.get('connection_string'))
            )
        else:
            validation['adls_valid'] = True  # Not required
        
        # Validate Key Vault
        if features.get('enable_keyvault'):
            kv_config = self.get_keyvault_config()
            validation['keyvault_valid'] = bool(kv_config.get('vault_url'))
        else:
            validation['keyvault_valid'] = True
        
        # Validate Azure ML
        if features.get('enable_aml'):
            aml_config = self.get_aml_config()
            validation['aml_valid'] = bool(
                aml_config.get('subscription_id') and
                aml_config.get('resource_group') and
                aml_config.get('workspace_name')
            )
        else:
            validation['aml_valid'] = True
        
        # Validate Databricks
        if features.get('enable_databricks'):
            db_config = self.get_databricks_config()
            validation['databricks_valid'] = bool(
                db_config.get('workspace_url') and
                db_config.get('access_token')
            )
        else:
            validation['databricks_valid'] = True
        
        # Validate Monitoring
        if features.get('enable_monitoring'):
            mon_config = self.get_monitoring_config()
            validation['monitoring_valid'] = bool(
                mon_config.get('connection_string') or
                mon_config.get('instrumentation_key')
            )
        else:
            validation['monitoring_valid'] = True
        
        validation['overall_valid'] = all(validation.values())
        return validation
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get complete pipeline configuration with Azure settings"""
        features = self.get_features()
        
        config = {
            'data_processor': {
                'missing_value_strategy': 'auto',
                'outlier_method': 'iqr',
                'scaling_method': 'robust',
                'use_adls': features.get('enable_adls', False)
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
                'enable_mlflow': not features.get('enable_aml', False),  # Use MLflow if AML not enabled
                'enable_aml': features.get('enable_aml', False),
                'use_databricks': features.get('enable_databricks', False),
                'experiment_name': 'dynamic_pricing_pipeline'
            },
            'azure': {
                'adls': self.get_adls_config() if features.get('enable_adls') else {},
                'keyvault': self.get_keyvault_config() if features.get('enable_keyvault') else {},
                'aml': self.get_aml_config() if features.get('enable_aml') else {},
                'databricks': self.get_databricks_config() if features.get('enable_databricks') else {},
                'monitoring': self.get_monitoring_config() if features.get('enable_monitoring') else {},
                'features': features
            }
        }
        
        return config


def get_azure_config() -> AzureConfig:
    """Get Azure configuration instance"""
    return AzureConfig(use_env_variables=True)


def create_azure_pipeline_config() -> Dict[str, Any]:
    """Create pipeline configuration with Azure integration"""
    azure_config = get_azure_config()
    return azure_config.get_pipeline_config()


def validate_azure_setup() -> Dict[str, Any]:
    """Validate Azure setup and return status"""
    azure_config = get_azure_config()
    
    validation = azure_config.validate_config()
    features = azure_config.get_features()
    
    status = {
        'azure_enabled': azure_config.is_azure_enabled(),
        'validation': validation,
        'enabled_features': [k for k, v in features.items() if v],
        'disabled_features': [k for k, v in features.items() if not v],
        'recommendations': []
    }
    
    # Add recommendations
    if not validation['overall_valid']:
        status['recommendations'].append("Some Azure services are misconfigured")
    
    if not features.get('enable_monitoring'):
        status['recommendations'].append("Consider enabling Azure Monitor for better observability")
    
    if not features.get('enable_adls'):
        status['recommendations'].append("Consider enabling ADLS for scalable data storage")
    
    return status
