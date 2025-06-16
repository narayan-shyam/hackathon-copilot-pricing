# Login
az login

az account set --subscription TPL-2025

#  Quick Resource Discovery Script:

echo "=== ALL RESOURCES WITH 'oops' IN CENTRAL INDIA ==="
az resource list \
  --query "[?contains(name, 'oops') && location=='centralindia'].{Name:name, Type:type, ResourceGroup:resourceGroup}" \
  -o table

echo -e "\n=== STORAGE ACCOUNTS ==="
az storage account list \
  --query "[?contains(name, 'oops') && location=='centralindia'].{Name:name, ResourceGroup:resourceGroup}" \
  -o table

echo -e "\n=== KEY VAULTS ==="
az keyvault list \
  --query "[?contains(name, 'oops') && location=='centralindia'].{Name:name, ResourceGroup:resourceGroup}" \
  -o table

echo -e "\n=== DATABRICKS WORKSPACES ==="
az resource list \
  --resource-type "Microsoft.Databricks/workspaces" \
  --query "[?contains(name, 'oops') && location=='centralindia'].{Name:name, ResourceGroup:resourceGroup}" \
  -o table

echo -e "\n=== MACHINE LEARNING WORKSPACES ==="
az resource list \
  --resource-type "Microsoft.MachineLearningServices/workspaces" \
  --query "[?contains(name, 'oops') && location=='centralindia'].{Name:name, ResourceGroup:resourceGroup}" \
  -o table


# Connectivity Test Plan

# Get storage account key first
az storage account keys list \
  --resource-group tpl-oops-all-ai \
  --account-name oopsallaiadls \
  --query '[0].value' -o tsv

  output=y1gFhRFqb9XnP0SwGSn3bdF+Hdtl/xBctKIfuSguwzRKEhzPjNqLTsXWtj+q+piLchMpN0wLMz+a+AStuIGDCA==

# Store this key in Key Vault:
# Store storage key in Key Vault (replace YOUR_STORAGE_KEY with actual key)
az keyvault secret set \
  --vault-name oopsallai-kv \
  --name "storage-account-key" \
  --value "y1gFhRFqb9XnP0SwGSn3bdF+Hdtl/xBctKIfuSguwzRKEhzPjNqLTsXWtj+q+piLchMpN0wLMz+a+AStuIGDCA==" \
  --subscription TPL-2025