"""
Azure Databricks Manager
Distributed processing and training integration
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import json
import time
from datetime import datetime

# Databricks imports with error handling
try:
    from databricks import sql
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.jobs import JobSettings, Task, NotebookTask
    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False

# PySpark imports for data processing
try:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql.functions import col, avg, sum as spark_sum, count, when
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatabricksManager:
    """Azure Databricks integration for distributed processing"""
    
    def __init__(self, server_hostname: str = None, http_path: str = None, 
                 access_token: str = None, workspace_url: str = None):
        
        if not DATABRICKS_AVAILABLE:
            logger.warning("Databricks SDK not available. Install with: pip install databricks-sql-connector")
            self.connection = None
            self.workspace_client = None
            return
        
        self.server_hostname = server_hostname
        self.http_path = http_path
        self.access_token = access_token
        self.workspace_url = workspace_url
        self.connection = None
        self.workspace_client = None
        
        # Initialize connections
        self._initialize_sql_connection()
        self._initialize_workspace_client()
    
    def _initialize_sql_connection(self):
        """Initialize SQL connection to Databricks"""
        if not all([self.server_hostname, self.http_path, self.access_token]):
            logger.warning("Missing Databricks SQL connection parameters")
            return
        
        try:
            self.connection = sql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token
            )
            logger.info("Databricks SQL connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Databricks SQL: {e}")
            self.connection = None
    
    def _initialize_workspace_client(self):
        """Initialize workspace client for jobs and clusters"""
        if not self.workspace_url or not self.access_token:
            logger.warning("Missing Databricks workspace parameters")
            return
        
        try:
            self.workspace_client = WorkspaceClient(
                host=self.workspace_url,
                token=self.access_token
            )
            logger.info("Databricks workspace client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize workspace client: {e}")
            self.workspace_client = None
    
    def execute_sql_query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Execute SQL query on Databricks"""
        if not self.connection:
            logger.error("No Databricks SQL connection available")
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            
            # Fetch results
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            logger.info(f"Executed SQL query, returned {len(results)} rows")
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute SQL query: {e}")
            return None
    
    def submit_training_job(self, notebook_path: str, cluster_id: str, 
                           dataset_table: str, target_column: str) -> Optional[str]:
        """Submit distributed training job to Databricks"""
        parameters = {
            "dataset_table": dataset_table,
            "target_column": target_column,
            "timestamp": datetime.now().isoformat()
        }
        
        job_name = f"pricing_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self.submit_notebook_job(
            notebook_path=notebook_path,
            cluster_id=cluster_id,
            job_name=job_name,
            parameters=parameters
        )
