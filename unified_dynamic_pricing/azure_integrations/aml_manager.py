"""
Azure Machine Learning Manager
Enterprise ML lifecycle management integration
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import joblib

# Azure ML imports with error handling
try:
    from azureml.core import Workspace, Experiment, Run, Model
    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.core.environment import Environment
    from azureml.core.model import InferenceConfig
    from azureml.core.webservice import AciWebservice
    from azureml.train.sklearn import SKLearn
    from azureml.core.runconfig import RunConfiguration
    from azureml.core.authentication import ServicePrincipalAuthentication
    AZUREML_AVAILABLE = True
except ImportError:
    AZUREML_AVAILABLE = False

logger = logging.getLogger(__name__)

class AzureMLManager:
    """Azure Machine Learning integration for the pricing pipeline"""
    
    def __init__(self, subscription_id: str = None, resource_group: str = None, 
                 workspace_name: str = None, config_file: str = None):
        
        if not AZUREML_AVAILABLE:
            logger.warning("Azure ML SDK not available. Install with: pip install azureml-core")
            self.workspace = None
            return
        
        try:
            # Initialize workspace
            if config_file and os.path.exists(config_file):
                self.workspace = Workspace.from_config(path=config_file)
            elif all([subscription_id, resource_group, workspace_name]):
                self.workspace = Workspace(
                    subscription_id=subscription_id,
                    resource_group=resource_group,
                    workspace_name=workspace_name
                )
            else:
                # Try to get from default config
                self.workspace = Workspace.from_config()
            
            self.experiment_name = "dynamic-pricing-pipeline"
            self.experiment = Experiment(workspace=self.workspace, name=self.experiment_name)
            
            logger.info(f"Azure ML workspace initialized: {self.workspace.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure ML workspace: {e}")
            self.workspace = None
            self.experiment = None
    
    def create_experiment(self, experiment_name: str) -> Optional[Experiment]:
        """Create or get existing experiment"""
        if not self.workspace:
            logger.error("Azure ML workspace not initialized")
            return None
        
        try:
            experiment = Experiment(workspace=self.workspace, name=experiment_name)
            logger.info(f"Created/Retrieved experiment: {experiment_name}")
            return experiment
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return None
    
    def start_run(self, experiment_name: str = None, run_name: str = None) -> Optional[Run]:
        """Start a new ML run"""
        if not self.workspace:
            logger.error("Azure ML workspace not initialized")
            return None
        
        try:
            experiment = self.experiment
            if experiment_name:
                experiment = self.create_experiment(experiment_name)
            
            if not experiment:
                return None
            
            # Generate run name if not provided
            if not run_name:
                run_name = f"pricing_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            run = experiment.start_logging(display_name=run_name)
            logger.info(f"Started run: {run_name}")
            return run
            
        except Exception as e:
            logger.error(f"Failed to start run: {e}")
            return None
    
    def log_metrics(self, run: Run, metrics: Dict[str, float]):
        """Log metrics to Azure ML run"""
        if not run:
            logger.error("No active run provided")
            return
        
        try:
            for metric_name, metric_value in metrics.items():
                run.log(metric_name, metric_value)
            
            logger.info(f"Logged {len(metrics)} metrics to run")
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_parameters(self, run: Run, parameters: Dict[str, Any]):
        """Log parameters to Azure ML run"""
        if not run:
            logger.error("No active run provided")
            return
        
        try:
            for param_name, param_value in parameters.items():
                # Convert complex objects to strings
                if isinstance(param_value, (dict, list)):
                    param_value = json.dumps(param_value)
                
                run.log(param_name, param_value)
            
            logger.info(f"Logged {len(parameters)} parameters to run")
            
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def register_model(self, run: Run, model_name: str, model_path: str, 
                      model_description: str = None, tags: Dict[str, str] = None,
                      model_framework: str = "sklearn") -> Optional[Model]:
        """Register model to Azure ML model registry"""
        if not run or not self.workspace:
            logger.error("No active run or workspace")
            return None
        
        try:
            # Default tags
            default_tags = {
                'framework': model_framework,
                'type': 'pricing_model',
                'created_date': datetime.now().isoformat()
            }
            
            if tags:
                default_tags.update(tags)
            
            model = run.register_model(
                model_name=model_name,
                model_path=model_path,
                description=model_description or f"Dynamic pricing model - {model_name}",
                tags=default_tags
            )
            
            logger.info(f"Registered model: {model_name} (version {model.version})")
            return model
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def deploy_model(self, model: Model, service_name: str, 
                    deployment_config: Dict[str, Any] = None) -> Optional[Any]:
        """Deploy model as web service"""
        if not model or not self.workspace:
            logger.error("No model or workspace provided")
            return None
        
        try:
            # Default deployment configuration
            default_config = {
                'cpu_cores': 1,
                'memory_gb': 1,
                'enable_app_insights': True,
                'auth_enabled': True
            }
            
            if deployment_config:
                default_config.update(deployment_config)
            
            # Create deployment configuration
            aci_config = AciWebservice.deploy_configuration(**default_config)
            
            # Create inference configuration
            environment = Environment.from_conda_specification(
                name="pricing-env",
                file_path="./environment.yml"
            )
            
            inference_config = InferenceConfig(
                entry_script="score.py",
                environment=environment
            )
            
            # Deploy the service
            service = Model.deploy(
                workspace=self.workspace,
                name=service_name,
                models=[model],
                inference_config=inference_config,
                deployment_config=aci_config,
                overwrite=True
            )
            
            service.wait_for_deployment(show_output=True)
            
            logger.info(f"Deployed model as service: {service_name}")
            logger.info(f"Service URL: {service.scoring_uri}")
            
            return service
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return None
    
    def get_model(self, model_name: str, version: int = None) -> Optional[Model]:
        """Get model from registry"""
        if not self.workspace:
            logger.error("Azure ML workspace not initialized")
            return None
        
        try:
            if version:
                model = Model(workspace=self.workspace, name=model_name, version=version)
            else:
                model = Model(workspace=self.workspace, name=model_name)
            
            logger.info(f"Retrieved model: {model_name} (version {model.version})")
            return model
            
        except Exception as e:
            logger.error(f"Failed to get model {model_name}: {e}")
            return None
    
    def list_models(self, model_name: str = None, tags: Dict[str, str] = None) -> List[Model]:
        """List models in registry"""
        if not self.workspace:
            logger.error("Azure ML workspace not initialized")
            return []
        
        try:
            models = Model.list(
                workspace=self.workspace,
                name=model_name,
                tags=tags
            )
            
            logger.info(f"Found {len(models)} models")
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def create_compute_target(self, compute_name: str, vm_size: str = "STANDARD_D2_V2",
                             min_nodes: int = 0, max_nodes: int = 4) -> Optional[ComputeTarget]:
        """Create or get compute target"""
        if not self.workspace:
            logger.error("Azure ML workspace not initialized")
            return None
        
        try:
            # Check if compute target already exists
            try:
                compute_target = ComputeTarget(workspace=self.workspace, name=compute_name)
                logger.info(f"Using existing compute target: {compute_name}")
                return compute_target
            except:
                pass
            
            # Create new compute target
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                min_nodes=min_nodes,
                max_nodes=max_nodes
            )
            
            compute_target = ComputeTarget.create(
                workspace=self.workspace,
                name=compute_name,
                provisioning_configuration=compute_config
            )
            
            compute_target.wait_for_completion(show_output=True)
            
            logger.info(f"Created compute target: {compute_name}")
            return compute_target
            
        except Exception as e:
            logger.error(f"Failed to create compute target: {e}")
            return None
    
    def submit_training_job(self, script_path: str, compute_target_name: str,
                           parameters: Dict[str, Any] = None) -> Optional[Run]:
        """Submit training job to compute target"""
        if not self.workspace:
            logger.error("Azure ML workspace not initialized")
            return None
        
        try:
            # Get compute target
            compute_target = ComputeTarget(workspace=self.workspace, name=compute_target_name)
            
            # Create run configuration
            run_config = RunConfiguration()
            run_config.target = compute_target
            
            # Create SKLearn estimator
            estimator = SKLearn(
                source_directory='.',
                script_params=parameters or {},
                compute_target=compute_target,
                entry_script=script_path
            )
            
            # Submit the job
            run = self.experiment.submit(estimator)
            
            logger.info(f"Submitted training job: {run.id}")
            return run
            
        except Exception as e:
            logger.error(f"Failed to submit training job: {e}")
            return None
    
    def monitor_run(self, run: Run, wait_for_completion: bool = False):
        """Monitor run progress"""
        if not run:
            logger.error("No run provided")
            return
        
        try:
            if wait_for_completion:
                run.wait_for_completion(show_output=True)
            
            # Get run status
            status = run.get_status()
            logger.info(f"Run status: {status}")
            
            # Get run metrics
            metrics = run.get_metrics()
            if metrics:
                logger.info("Run metrics:")
                for metric_name, metric_value in metrics.items():
                    logger.info(f"  {metric_name}: {metric_value}")
            
        except Exception as e:
            logger.error(f"Failed to monitor run: {e}")
    
    def get_run_history(self, experiment_name: str = None, limit: int = 10) -> List[Run]:
        """Get run history for experiment"""
        if not self.workspace:
            logger.error("Azure ML workspace not initialized")
            return []
        
        try:
            experiment = self.experiment
            if experiment_name:
                experiment = Experiment(workspace=self.workspace, name=experiment_name)
            
            runs = list(experiment.get_runs())[:limit]
            
            logger.info(f"Retrieved {len(runs)} runs from experiment")
            return runs
            
        except Exception as e:
            logger.error(f"Failed to get run history: {e}")
            return []
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs"""
        if not self.workspace:
            logger.error("Azure ML workspace not initialized")
            return {}
        
        try:
            comparison_data = {}
            
            for run_id in run_ids:
                try:
                    run = Run(experiment=self.experiment, run_id=run_id)
                    metrics = run.get_metrics()
                    comparison_data[run_id] = {
                        'status': run.get_status(),
                        'metrics': metrics,
                        'start_time': run.get_details().get('startTimeUtc'),
                        'end_time': run.get_details().get('endTimeUtc')
                    }
                except Exception as e:
                    logger.warning(f"Failed to get data for run {run_id}: {e}")
            
            logger.info(f"Compared {len(comparison_data)} runs")
            return comparison_data
            
        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return {}
    
    def create_environment_file(self, file_path: str = "./environment.yml"):
        """Create environment file for deployment"""
        try:
            environment_content = """
name: pricing-env
dependencies:
  - python=3.8
  - scikit-learn
  - pandas
  - numpy
  - joblib
  - pip
  - pip:
    - azureml-defaults
    - azure-storage-file-datalake
    - azure-keyvault-secrets
    - azure-identity
"""
            
            with open(file_path, 'w') as f:
                f.write(environment_content.strip())
            
            logger.info(f"Created environment file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create environment file: {e}")
            return False
    
    def create_scoring_script(self, file_path: str = "./score.py"):
        """Create scoring script for model deployment"""
        try:
            scoring_script = '''
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('pricing-model')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Return predictions as JSON
        return json.dumps({
            'predictions': predictions.tolist(),
            'status': 'success'
        })
    except Exception as e:
        return json.dumps({
            'error': str(e),
            'status': 'error'
        })
'''
            
            with open(file_path, 'w') as f:
                f.write(scoring_script.strip())
            
            logger.info(f"Created scoring script: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create scoring script: {e}")
            return False


# Azure ML integration for the pricing pipeline
class AzureMLPricingPipeline:
    """Integration class for pricing pipeline with Azure ML"""
    
    def __init__(self, aml_manager: AzureMLManager):
        self.aml_manager = aml_manager
        self.current_run = None
    
    def train_with_aml_tracking(self, model_trainer, X_train, y_train, 
                               experiment_name: str = None):
        """Train models with Azure ML tracking"""
        if not self.aml_manager.workspace:
            logger.warning("Azure ML not available, falling back to local training")
            return model_trainer.train_all_models(X_train, y_train)
        
        try:
            # Start Azure ML run
            self.current_run = self.aml_manager.start_run(experiment_name)
            
            if not self.current_run:
                logger.warning("Failed to start Azure ML run, falling back to local training")
                return model_trainer.train_all_models(X_train, y_train)
            
            # Log dataset information
            self.aml_manager.log_parameters(self.current_run, {
                'training_samples': len(X_train),
                'features': len(X_train.columns),
                'target_mean': float(y_train.mean()),
                'target_std': float(y_train.std())
            })
            
            # Train models
            training_results = model_trainer.train_all_models(X_train, y_train)
            
            # Log model results to Azure ML
            for model_name, model_info in training_results.items():
                metrics = {
                    f'{model_name}_cv_score': model_info['cv_score']
                }
                self.aml_manager.log_metrics(self.current_run, metrics)
                self.aml_manager.log_parameters(self.current_run, {
                    f'{model_name}_params': model_info['best_params']
                })
            
            # Log best model information
            if model_trainer.best_model:
                best_model_metrics = {
                    'best_model_cv_score': model_trainer.best_model['cv_score']
                }
                self.aml_manager.log_metrics(self.current_run, best_model_metrics)
                self.aml_manager.log_parameters(self.current_run, {
                    'best_model_name': model_trainer.best_model['name'],
                    'best_model_params': model_trainer.best_model['params']
                })
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error in Azure ML training: {e}")
            return model_trainer.train_all_models(X_train, y_train)
    
    def register_best_model(self, model_trainer, model_name: str = "pricing-model"):
        """Register best model to Azure ML registry"""
        if not self.current_run or not model_trainer.best_model:
            logger.warning("No active run or best model to register")
            return None
        
        try:
            # Save model locally first
            model_path = f"./{model_name}.joblib"
            joblib.dump(model_trainer.best_model['model'], model_path)
            
            # Upload model file
            self.current_run.upload_file("outputs/" + model_path, model_path)
            
            # Register model
            tags = {
                'model_type': model_trainer.best_model['name'],
                'cv_score': str(model_trainer.best_model['cv_score']),
                'training_date': datetime.now().isoformat()
            }
            
            model = self.aml_manager.register_model(
                run=self.current_run,
                model_name=model_name,
                model_path="outputs/" + model_path,
                tags=tags
            )
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def complete_run(self):
        """Complete the current Azure ML run"""
        if self.current_run:
            try:
                self.current_run.complete()
                logger.info("Azure ML run completed")
            except Exception as e:
                logger.error(f"Failed to complete run: {e}")
            finally:
                self.current_run = None
