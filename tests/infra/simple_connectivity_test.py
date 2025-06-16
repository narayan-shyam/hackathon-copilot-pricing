#!/usr/bin/env python3
"""
Simple Azure Connectivity Test - No Azure CLI Required
Tests basic connectivity using direct credentials
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_storage_with_key():
    """Test storage account using account key directly"""
    try:
        from azure.storage.blob import BlobServiceClient
        
        storage_account_name = "oopsallaiadls"
        storage_account_key = "y1gFhRFqb9XnP0SwGSn3bdF+Hdtl/xBctKIfuSguwzRKEhzPjNqLTsXWtj+q+piLchMpN0wLMz+a+AStuIGDCA=="
        container_name = "pricing-data"
        
        blob_service_client = BlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=storage_account_key
        )
        
        # Test container access
        container_client = blob_service_client.get_container_client(container_name)
        
        try:
            blobs = list(container_client.list_blobs())
            blob_count = len(blobs)
        except Exception as e:
            if "ContainerNotFound" in str(e):
                # Create container
                container_client = blob_service_client.create_container(container_name)
                blobs = list(container_client.list_blobs())
                blob_count = len(blobs)
                print("  📁 Created container 'pricing-data'")
            else:
                raise e
        
        # Test write access
        test_blob_name = "connectivity-test.txt"
        test_content = "Azure connectivity test successful!"
        
        blob_client = container_client.get_blob_client(test_blob_name)
        blob_client.upload_blob(test_content, overwrite=True)
        
        # Test read access
        downloaded_content = blob_client.download_blob().readall().decode('utf-8')
        
        return {
            'status': 'success',
            'message': f'✅ ADLS connected - {blob_count} blobs in container',
            'test_write': 'success',
            'test_read': downloaded_content == test_content
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f'❌ ADLS connection failed: {str(e)}'
        }

def test_databricks_api():
    """Test Databricks API connectivity"""
    try:
        workspace_url = "https://adb-1210314903119401.1.azuredatabricks.net"
        access_token = "dapif104ca64ec28177f6501a1c3efe1f441"
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        # Test API access
        response = requests.get(
            f"{workspace_url}/api/2.0/clusters/list",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            clusters = response.json().get('clusters', [])
            return {
                'status': 'success',
                'message': f'✅ Databricks API connected - {len(clusters)} clusters',
                'cluster_count': len(clusters)
            }
        else:
            return {
                'status': 'error',
                'message': f'❌ Databricks API failed: {response.status_code}'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': f'❌ Databricks API failed: {str(e)}'
        }

def test_key_vault_direct():
    """Test Key Vault without Azure CLI"""
    try:
        # This would require proper authentication setup
        # For now, just check if we can construct the URL
        vault_url = "https://oopsallai-kv.vault.azure.net/"
        
        # Test basic connectivity (this will fail auth but confirms URL is reachable)
        response = requests.get(vault_url, timeout=10)
        
        if response.status_code in [401, 403]:  # Auth required but service is reachable
            return {
                'status': 'success',
                'message': '✅ Key Vault is reachable (authentication required)',
                'note': 'Authentication will be handled via Databricks secret scope'
            }
        else:
            return {
                'status': 'warning',
                'message': f'⚠️ Key Vault response: {response.status_code}'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': f'❌ Key Vault test failed: {str(e)}'
        }

def run_simple_test():
    """Run simple connectivity test"""
    print("🚀 Simple Azure Connectivity Test (No CLI Required)")
    print("=" * 60)
    
    tests = {
        'Azure Data Lake Storage': test_storage_with_key,
        'Databricks API': test_databricks_api,
        'Key Vault Reachability': test_key_vault_direct
    }
    
    results = {}
    
    for service_name, test_func in tests.items():
        print(f"\n🧪 Testing {service_name}...")
        try:
            result = test_func()
            results[service_name] = result
            print(f"  {result['message']}")
            
            # Print additional details
            for key, value in result.items():
                if key not in ['status', 'message']:
                    print(f"  • {key}: {value}")
                    
        except Exception as e:
            results[service_name] = {
                'status': 'error',
                'message': f'❌ Test failed: {str(e)}'
            }
            print(f"  ❌ Test failed: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SIMPLE CONNECTIVITY TEST SUMMARY")
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
    
    print(f"\nOverall Status: {success_count}/{total_count} services working")
    
    if success_count >= 2:
        print("🎉 Core services are working! You can proceed with Databricks setup.")
        print("\n🔄 Next Steps:")
        print("1. Store secrets in Key Vault via Azure Portal")
        print("2. Create Databricks secret scope")
        print("3. Test from Databricks notebook")
    else:
        print("❌ Multiple issues detected. Check configurations.")
    
    return results

if __name__ == "__main__":
    # Check required packages
    try:
        from azure.storage.blob import BlobServiceClient
        import requests
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Install with: pip install azure-storage-blob requests")
        exit(1)
    
    results = run_simple_test()