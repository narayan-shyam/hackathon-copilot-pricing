"""
Azure ML Model Deployment Script for Dynamic Pricing Pipeline
This script handles the deployment of trained models from Databricks MLflow to Azure ML Managed Endpoints

Module 3: Model Deployment Automation
Ready for REST API testing and monitoring
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Azure ML imports
try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        Model, 
        ManagedOnlineEndpoint, 
        ManagedOnlineDeployment,
        Environment,
        CodeConfiguration
    )
    from azure.identity import DefaultAzureCredential
    from azure.core.exceptions import ResourceExistsError
    print("✅ Azure ML SDK v2 imported successfully")
except ImportError as e:
    print(f"❌ Azure ML SDK not found: {e}")
    print("💡 Install with: pip install azure-ai-ml azure-identity")

class AzureMLDynamicPricingDeployer:
    """
    Handles deployment of dynamic pricing models to Azure ML Managed Endpoints
    """
    
    def __init__(self, 
                 subscription_id: str,
                 resource_group: str,
                 workspace_name: str,
                 model_name: str = "dynamic_pricing_model",
                 endpoint_name: str = "dynamic-pricing-endpoint"):
        
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.model_name = model_name
        self.endpoint_name = endpoint_name
        
        # Initialize Azure ML client
        try:
            credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name
            )
            print(f"✅ Connected to Azure ML workspace: {workspace_name}")
        except Exception as e:
            print(f"❌ Failed to connect to Azure ML: {e}")
            raise
        
        # Setup logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'azure_ml_deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def register_model_from_databricks(self, 
                                     databricks_model_uri: str,
                                     model_version: str = "1",
                                     description: str = "Dynamic pricing model from Databricks") -> str:
        """
        Register model from Databricks MLflow to Azure ML
        
        Args:
            databricks_model_uri: MLflow model URI from Databricks
            model_version: Version of the model
            description: Model description
            
        Returns:
            Azure ML model name and version
        """
        self.logger.info(f"Registering model from Databricks: {databricks_model_uri}")
        
        try:
            # Create Azure ML Model entity
            model = Model(
                name=self.model_name,
                version=model_version,
                description=description,
                type="mlflow_model",
                path=databricks_model_uri,
                properties={
                    "source": "databricks_mlflow",
                    "deployment_ready": "true",
                    "created_date": datetime.now().isoformat()
                }
            )
            
            # Register the model
            registered_model = self.ml_client.models.create_or_update(model)
            
            self.logger.info(f"✅ Model registered: {registered_model.name}:{registered_model.version}")
            return f"{registered_model.name}:{registered_model.version}"
            
        except Exception as e:
            self.logger.error(f"❌ Model registration failed: {e}")
            raise
    
    def create_managed_endpoint(self, 
                              endpoint_description: str = "Dynamic Pricing REST API Endpoint") -> bool:
        """
        Create Azure ML Managed Online Endpoint
        
        Args:
            endpoint_description: Description for the endpoint
            
        Returns:
            Success status
        """
        self.logger.info(f"Creating managed endpoint: {self.endpoint_name}")
        
        try:
            # Create endpoint entity
            endpoint = ManagedOnlineEndpoint(
                name=self.endpoint_name,
                description=endpoint_description,
                auth_mode="key",
                tags={
                    "project": "dynamic_pricing",
                    "model": self.model_name,
                    "created_date": datetime.now().strftime("%Y-%m-%d"),
                    "environment": "production"
                }
            )
            
            # Create the endpoint
            endpoint_result = self.ml_client.online_endpoints.begin_create_or_update(endpoint)
            endpoint_result.wait()
            
            self.logger.info(f"✅ Endpoint created: {self.endpoint_name}")
            return True
            
        except ResourceExistsError:
            self.logger.warning(f"⚠️ Endpoint {self.endpoint_name} already exists")
            return True
        except Exception as e:
            self.logger.error(f"❌ Endpoint creation failed: {e}")
            raise
    
    def create_deployment(self,
                         model_version: str = "1",
                         instance_type: str = "Standard_DS3_v2",
                         instance_count: int = 1,
                         deployment_name: str = "blue") -> bool:
        """
        Deploy model to the managed endpoint
        
        Args:
            model_version: Version of the model to deploy
            instance_type: Azure VM instance type
            instance_count: Number of instances
            deployment_name: Name of the deployment
            
        Returns:
            Success status
        """
        self.logger.info(f"Creating deployment: {deployment_name}")
        
        try:
            # Create deployment entity
            deployment = ManagedOnlineDeployment(
                name=deployment_name,
                endpoint_name=self.endpoint_name,
                model=f"{self.model_name}:{model_version}",
                instance_type=instance_type,
                instance_count=instance_count,
                environment_variables={
                    "MODEL_NAME": self.model_name,
                    "MODEL_VERSION": model_version,
                    "DEPLOYMENT_DATE": datetime.now().isoformat()
                },
                tags={
                    "model_version": model_version,
                    "instance_type": instance_type,
                    "deployment_strategy": "blue_green"
                }
            )
            
            # Create the deployment
            deployment_result = self.ml_client.online_deployments.begin_create_or_update(deployment)
            deployment_result.wait()
            
            # Set traffic to 100% for this deployment
            endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)
            endpoint.traffic = {deployment_name: 100}
            self.ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
            
            self.logger.info(f"✅ Deployment created: {deployment_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Deployment failed: {e}")
            raise
    
    def test_endpoint(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test the deployed endpoint with sample data
        
        Args:
            sample_data: Sample input data for testing
            
        Returns:
            Prediction response
        """
        self.logger.info("Testing endpoint with sample data")
        
        try:
            # Convert sample data to JSON
            request_data = json.dumps(sample_data)
            
            # Invoke the endpoint
            response = self.ml_client.online_endpoints.invoke(
                endpoint_name=self.endpoint_name,
                request_file=None,
                deployment_name="blue"
            )
            
            self.logger.info(f"✅ Endpoint test successful")
            return json.loads(response)
            
        except Exception as e:
            self.logger.error(f"❌ Endpoint test failed: {e}")
            raise
    
    def get_endpoint_info(self) -> Dict[str, Any]:
        """
        Get endpoint information including scoring URI and keys
        
        Returns:
            Endpoint information
        """
        try:
            endpoint = self.ml_client.online_endpoints.get(self.endpoint_name)
            keys = self.ml_client.online_endpoints.get_keys(self.endpoint_name)
            
            endpoint_info = {
                "endpoint_name": endpoint.name,
                "scoring_uri": endpoint.scoring_uri,
                "swagger_uri": endpoint.openapi_uri,
                "primary_key": keys.primary_key,
                "secondary_key": keys.secondary_key,
                "provisioning_state": endpoint.provisioning_state,
                "auth_mode": endpoint.auth_mode,
                "location": endpoint.location,
                "tags": endpoint.tags
            }
            
            self.logger.info(f"✅ Retrieved endpoint info: {self.endpoint_name}")
            return endpoint_info
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get endpoint info: {e}")
            raise
    
    def deploy_complete_pipeline(self,
                                databricks_model_uri: str,
                                model_version: str = "1",
                                sample_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete deployment pipeline: register model, create endpoint, deploy, and test
        
        Args:
            databricks_model_uri: MLflow model URI from Databricks
            model_version: Version of the model
            sample_data: Sample data for testing
            
        Returns:
            Complete deployment information
        """
        self.logger.info("🚀 Starting complete deployment pipeline")
        
        try:
            # Step 1: Register model from Databricks
            model_ref = self.register_model_from_databricks(
                databricks_model_uri=databricks_model_uri,
                model_version=model_version
            )
            
            # Step 2: Create managed endpoint
            self.create_managed_endpoint()
            
            # Step 3: Create deployment
            self.create_deployment(model_version=model_version)
            
            # Step 4: Get endpoint information
            endpoint_info = self.get_endpoint_info()
            
            # Step 5: Test endpoint if sample data provided
            test_result = None
            if sample_data:
                test_result = self.test_endpoint(sample_data)
            
            deployment_summary = {
                "status": "success",
                "model_reference": model_ref,
                "endpoint_info": endpoint_info,
                "test_result": test_result,
                "deployment_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info("🎉 Complete deployment pipeline finished successfully!")
            return deployment_summary
            
        except Exception as e:
            self.logger.error(f"❌ Deployment pipeline failed: {e}")
            raise

def main():
    """
    Main deployment function - ready for Module 3 execution
    """
    print("🔥 Azure ML Dynamic Pricing Model Deployment")
    print("=" * 50)
    
    # Configuration from environment variables
    subscription_id = os.getenv("AZURE_ML_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_ML_RESOURCE_GROUP", "pricing-rg")
    workspace_name = os.getenv("AZURE_ML_WORKSPACE_NAME", "pricing-ml-workspace")
    model_name = os.getenv("MODEL_NAME", "dynamic_pricing_model")
    
    if not subscription_id:
        print("❌ AZURE_ML_SUBSCRIPTION_ID environment variable not set")
        return
    
    # Sample configuration for testing
    databricks_model_uri = "models:/dynamic_pricing_model/1"  # Update with actual URI
    sample_data = {
        "base_price": 100.0,
        "cost": 60.0,
        "competitor_price": 95.0,
        "demand": 150,
        "inventory_level": 500,
        "customer_engagement": 0.75,
        "market_demand_factor": 1.2,
        "seasonal_factor": 1.1,
        "price_to_cost_ratio": 1.67,
        "profit_margin": 0.4,
        "day_of_week": 3,
        "month": 6,
        "quarter": 2,
        "is_weekend": 0
    }
    
    try:
        # Initialize deployer
        deployer = AzureMLDynamicPricingDeployer(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name,
            model_name=model_name
        )
        
        # Run complete deployment
        deployment_result = deployer.deploy_complete_pipeline(
            databricks_model_uri=databricks_model_uri,
            model_version="1",
            sample_data=sample_data
        )
        
        print("\n🎉 DEPLOYMENT SUCCESSFUL!")
        print(f"📡 Scoring URI: {deployment_result['endpoint_info']['scoring_uri']}")
        print(f"🔑 Primary Key: {deployment_result['endpoint_info']['primary_key'][:20]}...")
        print(f"📊 Test Result: {deployment_result['test_result']}")
        
        # Save deployment info for later use
        with open("deployment_info.json", "w") as f:
            json.dump(deployment_result, f, indent=2, default=str)
        
        print("\n✅ Deployment info saved to deployment_info.json")
        print("🚀 Ready for Module 4: Testing Framework Implementation!")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("💡 Check your Azure credentials and configuration")

if __name__ == "__main__":
    main()
