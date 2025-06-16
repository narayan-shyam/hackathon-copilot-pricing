"""
Enhanced Azure Key Vault Manager
Secure configuration and secrets management for the pricing pipeline
"""

from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
import logging
from typing import Dict, Any, Optional
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AzureKeyVaultManager:
    """Enhanced Key Vault manager for secure configuration and secrets"""
    
    def __init__(self, vault_url: str):
        self.vault_url = vault_url
        
        try:
            # Initialize Key Vault client with managed identity
            self.credential = DefaultAzureCredential()
            self.client = SecretClient(vault_url=vault_url, credential=self.credential)
            
            logger.info(f"Key Vault Manager initialized for: {vault_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Key Vault Manager: {e}")
            self.client = None
    
    def get_secret(self, secret_name: str, default_value: str = None) -> Optional[str]:
        """Get secret value from Key Vault"""
        if self.client is None:
            logger.error("Key Vault client not initialized")
            return default_value
            
        try:
            secret = self.client.get_secret(secret_name)
            logger.debug(f"Successfully retrieved secret: {secret_name}")
            return secret.value
            
        except Exception as e:
            logger.error(f"Failed to get secret {secret_name}: {e}")
            return default_value
    
    def set_secret(self, secret_name: str, secret_value: str, 
                   expires_on: datetime = None, tags: Dict[str, str] = None) -> bool:
        """Set secret value in Key Vault"""
        if self.client is None:
            logger.error("Key Vault client not initialized")
            return False
            
        try:
            self.client.set_secret(
                name=secret_name,
                value=secret_value,
                expires_on=expires_on,
                tags=tags
            )
            logger.info(f"Successfully set secret: {secret_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set secret {secret_name}: {e}")
            return False
    
    def get_database_config(self, environment: str = "dev") -> Dict[str, str]:
        """Get database configuration for specific environment"""
        config = {}
        
        # Database connection secrets
        secrets_map = {
            'host': f'db-{environment}-host',
            'port': f'db-{environment}-port', 
            'database': f'db-{environment}-name',
            'username': f'db-{environment}-username',
            'password': f'db-{environment}-password'
        }
        
        for key, secret_name in secrets_map.items():
            value = self.get_secret(secret_name)
            if value:
                config[key] = value
        
        logger.info(f"Retrieved database config for environment: {environment}")
        return config
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get external API keys and tokens"""
        api_keys = {}
        
        # Common API keys for pricing data
        api_secret_names = [
            'external-pricing-api-key',
            'competitor-data-api-key',
            'market-data-api-key',
            'customer-analytics-api-key',
            'databricks-token',
            'azure-ml-api-key'
        ]
        
        for secret_name in api_secret_names:
            value = self.get_secret(secret_name)
            if value:
                api_keys[secret_name.replace('-', '_')] = value
        
        logger.info(f"Retrieved {len(api_keys)} API keys")
        return api_keys
    
    def get_azure_config(self) -> Dict[str, str]:
        """Get Azure service configurations"""
        azure_config = {}
        
        # Azure service configurations
        azure_secrets = {
            'storage_account_name': 'azure-storage-account-name',
            'storage_account_key': 'azure-storage-account-key',
            'adls_connection_string': 'azure-adls-connection-string',
            'databricks_workspace_url': 'databricks-workspace-url',
            'databricks_token': 'databricks-token',
            'aml_subscription_id': 'azure-ml-subscription-id',
            'aml_resource_group': 'azure-ml-resource-group',
            'aml_workspace_name': 'azure-ml-workspace-name',
            'application_insights_key': 'application-insights-key'
        }
        
        for key, secret_name in azure_secrets.items():
            value = self.get_secret(secret_name)
            if value:
                azure_config[key] = value
        
        logger.info(f"Retrieved Azure configuration with {len(azure_config)} settings")
        return azure_config
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration"""
        config_secret_name = f'model-config-{model_name}'
        config_json = self.get_secret(config_secret_name)
        
        if config_json:
            try:
                return json.loads(config_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse model config JSON: {e}")
                return {}
        
        return {}
    
    def save_model_config(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Save model configuration to Key Vault"""
        config_secret_name = f'model-config-{model_name}'
        config_json = json.dumps(config, indent=2)
        
        tags = {
            'type': 'model-config',
            'model': model_name,
            'updated': datetime.now().isoformat()
        }
        
        return self.set_secret(config_secret_name, config_json, tags=tags)
    
    def get_pipeline_config(self, pipeline_name: str = "dynamic-pricing") -> Dict[str, Any]:
        """Get complete pipeline configuration"""
        config_secret_name = f'pipeline-config-{pipeline_name}'
        config_json = self.get_secret(config_secret_name)
        
        if config_json:
            try:
                return json.loads(config_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse pipeline config JSON: {e}")
                return {}
        
        # Return default configuration if not found
        return self._get_default_pipeline_config()
    
    def save_pipeline_config(self, config: Dict[str, Any], 
                           pipeline_name: str = "dynamic-pricing") -> bool:
        """Save pipeline configuration to Key Vault"""
        config_secret_name = f'pipeline-config-{pipeline_name}'
        config_json = json.dumps(config, indent=2)
        
        tags = {
            'type': 'pipeline-config',
            'pipeline': pipeline_name,
            'updated': datetime.now().isoformat()
        }
        
        return self.set_secret(config_secret_name, config_json, tags=tags)
    
    def get_environment_variables(self, environment: str = "production") -> Dict[str, str]:
        """Get environment-specific variables"""
        env_vars = {}
        
        # Environment-specific secrets
        env_secret_names = [
            f'{environment}-log-level',
            f'{environment}-debug-mode',
            f'{environment}-cache-enabled',
            f'{environment}-monitoring-enabled',
            f'{environment}-feature-flags'
        ]
        
        for secret_name in env_secret_names:
            value = self.get_secret(secret_name)
            if value:
                # Convert secret name to environment variable format
                env_var_name = secret_name.replace('-', '_').upper()
                env_vars[env_var_name] = value
        
        logger.info(f"Retrieved {len(env_vars)} environment variables for: {environment}")
        return env_vars
    
    def rotate_api_keys(self, api_name: str, new_key: str, 
                       backup_current: bool = True) -> bool:
        """Rotate API keys with backup"""
        current_secret_name = f'{api_name}-api-key'
        backup_secret_name = f'{api_name}-api-key-backup'
        
        try:
            # Backup current key if requested
            if backup_current:
                current_key = self.get_secret(current_secret_name)
                if current_key:
                    backup_tags = {
                        'type': 'api-key-backup',
                        'original_name': current_secret_name,
                        'backed_up_at': datetime.now().isoformat()
                    }
                    self.set_secret(backup_secret_name, current_key, tags=backup_tags)
            
            # Set new key
            new_tags = {
                'type': 'api-key',
                'api_name': api_name,
                'rotated_at': datetime.now().isoformat()
            }
            
            success = self.set_secret(current_secret_name, new_key, tags=new_tags)
            
            if success:
                logger.info(f"Successfully rotated API key for: {api_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to rotate API key for {api_name}: {e}")
            return False
    
    def list_secrets(self, filter_type: str = None) -> list:
        """List secrets with optional filtering"""
        if self.client is None:
            logger.error("Key Vault client not initialized")
            return []
            
        try:
            secrets = []
            secret_properties = self.client.list_properties_of_secrets()
            
            for secret_property in secret_properties:
                secret_info = {
                    'name': secret_property.name,
                    'enabled': secret_property.enabled,
                    'created_on': secret_property.created_on,
                    'updated_on': secret_property.updated_on,
                    'expires_on': secret_property.expires_on,
                    'tags': secret_property.tags or {}
                }
                
                # Filter by type if specified
                if filter_type is None or secret_info['tags'].get('type') == filter_type:
                    secrets.append(secret_info)
            
            logger.info(f"Found {len(secrets)} secrets" + 
                       (f" of type '{filter_type}'" if filter_type else ""))
            return secrets
            
        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []
    
    def _get_default_pipeline_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
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
                'enable_mlflow': True,
                'experiment_name': 'dynamic_pricing_pipeline'
            },
            'azure': {
                'enable_adls': True,
                'enable_databricks': False,
                'enable_aml': True,
                'enable_monitoring': True
            }
        }


# Configuration helper class
class SecureConfigManager:
    """Helper class for managing secure configurations"""
    
    def __init__(self, vault_manager: AzureKeyVaultManager, environment: str = "production"):
        self.vault_manager = vault_manager
        self.environment = environment
        self._cached_config = {}
    
    def get_complete_config(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get complete configuration from Key Vault"""
        if use_cache and self._cached_config:
            return self._cached_config
        
        config = {
            'database': self.vault_manager.get_database_config(self.environment),
            'api_keys': self.vault_manager.get_api_keys(),
            'azure': self.vault_manager.get_azure_config(),
            'pipeline': self.vault_manager.get_pipeline_config(),
            'environment': self.vault_manager.get_environment_variables(self.environment)
        }
        
        if use_cache:
            self._cached_config = config
        
        return config
    
    def refresh_config(self):
        """Refresh cached configuration"""
        self._cached_config = {}
        return self.get_complete_config(use_cache=False)
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate that required configuration is available"""
        config = self.get_complete_config()
        
        validation_results = {
            'database_config': bool(config.get('database', {})),
            'azure_storage': 'storage_account_name' in config.get('azure', {}),
            'databricks_config': 'databricks_workspace_url' in config.get('azure', {}),
            'aml_config': all(key in config.get('azure', {}) for key in 
                            ['aml_subscription_id', 'aml_resource_group', 'aml_workspace_name']),
            'monitoring_config': 'application_insights_key' in config.get('azure', {})
        }
        
        return validation_results
