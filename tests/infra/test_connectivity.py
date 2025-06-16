#!/usr/bin/env python3
"""
Azure Services Connectivity Test for Dynamic Pricing Pipeline
Tests connectivity between ADLS, AKV, AML, and ADB
"""

import os
import sys
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_azure_storage():
    """Test Azure Data Lake Storage connectivity"""
    try:
        from azure.storage.blob import BlobServiceClient
        from azure.identity import DefaultAzureCredential
        
        storage_account = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
        container_name = os.getenv('AZURE_ADLS_CONTAINER')
        
        # Try with Default Azure Credential first
        try:
            credential = DefaultAzureCredential()
            blob_client = BlobServiceClient(
                account_url=f"https://{storage_account}.blob.core.windows.net",
                credential=credential
            )
            container_client = blob_client.get_container_client(container_name)
            
            # Test container access
            blobs = list(container_client.list_blobs())
            return {
                'status': 'success',
                'method': 'managed_identity',
                'blob_count': len(blobs),
                'message': f'✅ ADLS connected with managed identity - {len(blobs)} blobs found'
            }
        except Exception as e:
            # Fallback to storage account key if available
            storage_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
            if storage_key:
                blob_client = BlobServiceClient(
                    account_url=f"https://{storage_account}.blob.core.windows.net",
                    credential=storage_key
                )
                container_client = blob_client.get_container_client(container_name)
                blobs = list(container_client.list_blobs())
                return {
                    'status': 'success',
                    'method': 'storage_key',
                    'blob_count': len(blobs),
                    'message': f'✅ ADLS connected with storage key - {len(blobs)} blobs found'
                }
            else:
                raise e
                
    except Exception as e:
        return {
            'status': 'error',
            'message': f'❌ ADLS connection failed: {str(e)}'
        }

def test_key_vault():
    """Test Azure Key Vault connectivity"""
    try:
        from azure.keyvault.secrets import SecretClient
        from azure.identity import DefaultAzureCredential
        
        vault_url = os.getenv('AZURE_KEYVAULT_URL')
        credential = DefaultAzureCredential()
        
        kv_client = SecretClient(
            vault_url=vault_url,
            credential=credential
        )
        
        # List secrets
        secrets = list(kv_client.list_properties_of_secrets())
        
        return {
            'status': 'success',
            'secret_count': len(secrets),
            'message': f'✅ Key Vault connected - {len(secrets)} secrets found'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'❌ Key Vault connection failed: {str(e)}'
        }

def test_azure_ml():
    """Test Azure Machine Learning connectivity"""
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential
        
        subscription_id = os.getenv('AZURE_ML_SUBSCRIPTION_ID')
        resource_group = os.getenv('AZURE_ML_RESOURCE_GROUP')
        workspace_name = os.getenv('AZURE_ML_WORKSPACE_NAME')
        
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        # Test workspace access
        workspace = ml_client.workspaces.get(workspace_name)
        
        # List datastores
        datastores = list(ml_client.datastores.list())
        
        return {
            'status': 'success',
            'workspace_name': workspace.name,
            'datastore_count': len(datastores),
            'message': f'✅ Azure ML connected - workspace: {workspace.name}, {len(datastores)} datastores'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'❌ Azure ML connection failed: {str(e)}'
        }

def test_databricks_api():
    """Test Databricks API connectivity"""
    try:
        import requests
        
        workspace_url = os.getenv('DATABRICKS_WORKSPACE_URL')
        access_token = os.getenv('DATABRICKS_ACCESS_TOKEN')
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Test API access - list clusters
        response = requests.get(
            f"{workspace_url}/api/2.0/clusters/list",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            clusters = response.json().get('clusters', [])
            return {
                'status': 'success',
                'cluster_count': len(clusters),
                'message': f'✅ Databricks API connected - {len(clusters)} clusters found'
            }
        else:
            return {
                'status': 'error',
                'message': f'❌ Databricks API failed: {response.status_code} - {response.text}'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': f'❌ Databricks API connection failed: {str(e)}'
        }

def test_application_insights():
    """Test Application Insights connectivity"""
    try:
        connection_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
        
        if not connection_string:
            return {
                'status': 'warning',
                'message': '⚠️ Application Insights connection string not configured'
            }
        
        # Parse connection string
        parts = dict(item.split('=', 1) for item in connection_string.split(';') if '=' in item)
        instrumentation_key = parts.get('InstrumentationKey')
        
        if instrumentation_key:
            return {
                'status': 'success',
                'instrumentation_key': f"{instrumentation_key[:8]}...",
                'message': f'✅ Application Insights configured - Key: {instrumentation_key[:8]}...'
            }
        else:
            return {
                'status': 'error',
                'message': '❌ Invalid Application Insights connection string'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': f'❌ Application Insights test failed: {str(e)}'
        }

def run_comprehensive_test():
    """Run comprehensive connectivity test"""
    print("🚀 Starting Azure Services Connectivity Test")
    print("=" * 60)
    
    # Check if .env is loaded
    print(f"Environment loaded: {bool(os.getenv('AZURE_STORAGE_ACCOUNT_NAME'))}")
    print()
    
    tests = {
        'Azure Data Lake Storage (ADLS)': test_azure_storage,
        'Azure Key Vault (AKV)': test_key_vault,
        'Azure Machine Learning (AML)': test_azure_ml,
        'Azure Databricks (ADB)': test_databricks_api,
        'Application Insights': test_application_insights
    }
    
    results = {}
    
    for service_name, test_func in tests.items():
        print(f"Testing {service_name}...")
        try:
            result = test_func()
            results[service_name] = result
            print(f"  {result['message']}")
        except Exception as e:
            results[service_name] = {
                'status': 'error',
                'message': f'❌ Test failed: {str(e)}'
            }
            print(f"  ❌ Test failed: {str(e)}")
        print()
    
    # Summary
    print("=" * 60)
    print("📊 CONNECTIVITY TEST SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_count = len(results)
    
    for service, result in results.items():
        status_emoji = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️'
        }.get(result['status'], '❓')
        
        print(f"{status_emoji} {service}: {result['status'].upper()}")
    
    print()
    print(f"Overall Status: {success_count}/{total_count} services connected successfully")
    
    if success_count == total_count:
        print("🎉 All services are connected! Your Azure setup is ready!")
    elif success_count >= 3:
        print("⚠️ Most services connected. Check failed connections above.")
    else:
        print("❌ Multiple connection failures. Review configuration and permissions.")
    
    return results

if __name__ == "__main__":
    # Install required packages if missing
    required_packages = [
        'azure-storage-blob',
        'azure-keyvault-secrets', 
        'azure-ai-ml',
        'azure-identity',
        'requests',
        'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '.').replace('_', '.'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    results = run_comprehensive_test()