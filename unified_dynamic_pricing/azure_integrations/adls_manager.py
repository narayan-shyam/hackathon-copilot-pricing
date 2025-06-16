"""
Azure Data Lake Storage Manager
Handles data storage and retrieval for the pricing pipeline
"""

import pandas as pd
import numpy as np
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential
import logging
from typing import Dict, Any, Optional, Union
import io
import json
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger(__name__)

class ADLSManager:
    """Azure Data Lake Storage manager for pricing pipeline data"""
    
    def __init__(self, storage_account_name: str, container_name: str = "pricing-data"):
        self.storage_account_name = storage_account_name
        self.container_name = container_name
        
        try:
            # Initialize ADLS client with managed identity
            self.credential = DefaultAzureCredential()
            self.service_client = DataLakeServiceClient(
                account_url=f"https://{storage_account_name}.dfs.core.windows.net",
                credential=self.credential
            )
            
            # Get container client
            self.file_system_client = self.service_client.get_file_system_client(
                file_system=container_name
            )
            
            # Ensure container exists
            self._ensure_container_exists()
            logger.info(f"ADLS Manager initialized for account: {storage_account_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ADLS Manager: {e}")
            self.service_client = None
            self.file_system_client = None
    
    def _ensure_container_exists(self):
        """Ensure the container exists, create if not"""
        try:
            self.file_system_client.get_file_system_properties()
        except Exception:
            try:
                self.file_system_client.create_file_system()
                logger.info(f"Created container: {self.container_name}")
            except Exception as e:
                logger.warning(f"Could not create container: {e}")
    
    def upload_dataframe(self, df: pd.DataFrame, file_path: str, 
                        format: str = "parquet", **kwargs) -> bool:
        """Upload pandas DataFrame to ADLS"""
        if self.file_system_client is None:
            logger.error("ADLS client not initialized")
            return False
            
        try:
            # Prepare data based on format
            if format.lower() == "parquet":
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False, **kwargs)
                data = buffer.getvalue()
            elif format.lower() == "csv":
                data = df.to_csv(index=False, **kwargs).encode('utf-8')
            elif format.lower() == "json":
                data = df.to_json(orient='records', **kwargs).encode('utf-8')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Upload to ADLS
            file_client = self.file_system_client.get_file_client(file_path)
            file_client.upload_data(data, overwrite=True)
            
            logger.info(f"Successfully uploaded DataFrame to ADLS: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame to ADLS: {e}")
            return False
    
    def download_dataframe(self, file_path: str, format: str = "parquet", **kwargs) -> Optional[pd.DataFrame]:
        """Download data from ADLS as pandas DataFrame"""
        if self.file_system_client is None:
            logger.error("ADLS client not initialized")
            return None
            
        try:
            file_client = self.file_system_client.get_file_client(file_path)
            download = file_client.download_file()
            data = download.readall()
            
            # Parse data based on format
            if format.lower() == "parquet":
                df = pd.read_parquet(io.BytesIO(data), **kwargs)
            elif format.lower() == "csv":
                df = pd.read_csv(io.StringIO(data.decode('utf-8')), **kwargs)
            elif format.lower() == "json":
                df = pd.read_json(io.StringIO(data.decode('utf-8')), **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Successfully downloaded DataFrame from ADLS: {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download DataFrame from ADLS: {e}")
            return None
    
    def upload_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """Upload JSON data to ADLS"""
        if self.file_system_client is None:
            logger.error("ADLS client not initialized")
            return False
            
        try:
            json_data = json.dumps(data, indent=2, default=str).encode('utf-8')
            file_client = self.file_system_client.get_file_client(file_path)
            file_client.upload_data(json_data, overwrite=True)
            
            logger.info(f"Successfully uploaded JSON to ADLS: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload JSON to ADLS: {e}")
            return False
    
    def download_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Download JSON data from ADLS"""
        if self.file_system_client is None:
            logger.error("ADLS client not initialized")
            return None
            
        try:
            file_client = self.file_system_client.get_file_client(file_path)
            download = file_client.download_file()
            data = json.loads(download.readall().decode('utf-8'))
            
            logger.info(f"Successfully downloaded JSON from ADLS: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to download JSON from ADLS: {e}")
            return None
    
    def list_files(self, directory_path: str = "") -> list:
        """List files in ADLS directory"""
        if self.file_system_client is None:
            logger.error("ADLS client not initialized")
            return []
            
        try:
            paths = self.file_system_client.get_paths(path=directory_path)
            file_list = [path.name for path in paths if not path.is_directory]
            
            logger.info(f"Found {len(file_list)} files in directory: {directory_path}")
            return file_list
            
        except Exception as e:
            logger.error(f"Failed to list files in ADLS: {e}")
            return []
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from ADLS"""
        if self.file_system_client is None:
            logger.error("ADLS client not initialized")
            return False
            
        try:
            file_client = self.file_system_client.get_file_client(file_path)
            file_client.delete_file()
            
            logger.info(f"Successfully deleted file from ADLS: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file from ADLS: {e}")
            return False
    
    def get_file_properties(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file properties from ADLS"""
        if self.file_system_client is None:
            logger.error("ADLS client not initialized")
            return None
            
        try:
            file_client = self.file_system_client.get_file_client(file_path)
            properties = file_client.get_file_properties()
            
            return {
                'name': file_path,
                'size': properties.size,
                'last_modified': properties.last_modified,
                'etag': properties.etag,
                'content_type': properties.content_settings.content_type if properties.content_settings else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get file properties from ADLS: {e}")
            return None
    
    def create_directory(self, directory_path: str) -> bool:
        """Create directory in ADLS"""
        if self.file_system_client is None:
            logger.error("ADLS client not initialized")
            return False
            
        try:
            directory_client = self.file_system_client.get_directory_client(directory_path)
            directory_client.create_directory()
            
            logger.info(f"Successfully created directory in ADLS: {directory_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directory in ADLS: {e}")
            return False
    
    def backup_data(self, source_path: str, backup_directory: str = "backups") -> bool:
        """Create backup of data with timestamp"""
        if self.file_system_client is None:
            logger.error("ADLS client not initialized")
            return False
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = source_path.split('/')[-1]
            backup_path = f"{backup_directory}/{timestamp}_{filename}"
            
            # Download source data
            source_client = self.file_system_client.get_file_client(source_path)
            download = source_client.download_file()
            data = download.readall()
            
            # Upload to backup location
            backup_client = self.file_system_client.get_file_client(backup_path)
            backup_client.upload_data(data, overwrite=True)
            
            logger.info(f"Successfully created backup: {source_path} -> {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False


# Data versioning utilities
class ADLSDataVersioning:
    """Handle data versioning in ADLS"""
    
    def __init__(self, adls_manager: ADLSManager, versioning_root: str = "versions"):
        self.adls_manager = adls_manager
        self.versioning_root = versioning_root
    
    def save_versioned_data(self, df: pd.DataFrame, dataset_name: str, 
                           version: str = None, metadata: Dict[str, Any] = None) -> str:
        """Save data with version control"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create version directory
        version_dir = f"{self.versioning_root}/{dataset_name}/{version}"
        data_path = f"{version_dir}/data.parquet"
        metadata_path = f"{version_dir}/metadata.json"
        
        # Save data
        success = self.adls_manager.upload_dataframe(df, data_path, format="parquet")
        
        if success and metadata:
            # Save metadata
            full_metadata = {
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                **metadata
            }
            self.adls_manager.upload_json(full_metadata, metadata_path)
        
        return version if success else None
    
    def load_versioned_data(self, dataset_name: str, version: str = "latest") -> Optional[pd.DataFrame]:
        """Load specific version of data"""
        if version == "latest":
            # Find latest version
            versions = self.list_versions(dataset_name)
            if not versions:
                return None
            version = max(versions)
        
        data_path = f"{self.versioning_root}/{dataset_name}/{version}/data.parquet"
        return self.adls_manager.download_dataframe(data_path, format="parquet")
    
    def list_versions(self, dataset_name: str) -> list:
        """List available versions for a dataset"""
        try:
            version_dir = f"{self.versioning_root}/{dataset_name}"
            files = self.adls_manager.list_files(version_dir)
            versions = list(set([f.split('/')[0] for f in files if '/' in f]))
            return sorted(versions)
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []
