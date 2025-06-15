"""
Custom exceptions for the unified dynamic pricing pipeline
"""

from typing import Optional, Dict, Any


class PipelineError(Exception):
    """Base exception for all pipeline-related errors"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (Context: {context_str})"
        return base_msg


class DataValidationError(PipelineError):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, validation_results: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="DATA_VALIDATION", context=validation_results)
        self.validation_results = validation_results


class DataProcessingError(PipelineError):
    """Raised when data preprocessing fails"""
    
    def __init__(self, message: str, processing_stage: Optional[str] = None, 
                 data_shape: Optional[tuple] = None):
        context = {}
        if processing_stage:
            context['stage'] = processing_stage
        if data_shape:
            context['data_shape'] = data_shape
        super().__init__(message, error_code="DATA_PROCESSING", context=context)


class FeatureEngineeringError(PipelineError):
    """Raised when feature engineering fails"""
    
    def __init__(self, message: str, feature_type: Optional[str] = None, 
                 feature_count: Optional[int] = None):
        context = {}
        if feature_type:
            context['feature_type'] = feature_type
        if feature_count:
            context['feature_count'] = feature_count
        super().__init__(message, error_code="FEATURE_ENGINEERING", context=context)


class ModelTrainingError(PipelineError):
    """Raised when model training fails"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 training_stage: Optional[str] = None):
        context = {}
        if model_name:
            context['model_name'] = model_name
        if training_stage:
            context['training_stage'] = training_stage
        super().__init__(message, error_code="MODEL_TRAINING", context=context)


class ModelEvaluationError(PipelineError):
    """Raised when model evaluation fails"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 metric_name: Optional[str] = None):
        context = {}
        if model_name:
            context['model_name'] = model_name
        if metric_name:
            context['metric_name'] = metric_name
        super().__init__(message, error_code="MODEL_EVALUATION", context=context)


class ConfigurationError(PipelineError):
    """Raised when configuration is invalid or missing"""
    
    def __init__(self, message: str, config_section: Optional[str] = None):
        context = {}
        if config_section:
            context['config_section'] = config_section
        super().__init__(message, error_code="CONFIGURATION", context=context)


class DependencyError(PipelineError):
    """Raised when required dependencies are missing"""
    
    def __init__(self, message: str, missing_package: Optional[str] = None, 
                 install_command: Optional[str] = None):
        context = {}
        if missing_package:
            context['missing_package'] = missing_package
        if install_command:
            context['install_command'] = install_command
        super().__init__(message, error_code="DEPENDENCY", context=context)


class PredictionError(PipelineError):
    """Raised when prediction fails"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 input_shape: Optional[tuple] = None):
        context = {}
        if model_name:
            context['model_name'] = model_name
        if input_shape:
            context['input_shape'] = input_shape
        super().__init__(message, error_code="PREDICTION", context=context)


class BusinessRuleViolationError(PipelineError):
    """Raised when business rules are violated"""
    
    def __init__(self, message: str, rule_name: Optional[str] = None, 
                 violation_count: Optional[int] = None):
        context = {}
        if rule_name:
            context['rule_name'] = rule_name
        if violation_count:
            context['violation_count'] = violation_count
        super().__init__(message, error_code="BUSINESS_RULE", context=context)


# Convenience functions for common error scenarios
def raise_data_validation_error(message: str, **kwargs):
    """Convenience function to raise data validation error"""
    raise DataValidationError(message, **kwargs)


def raise_model_training_error(message: str, model_name: str = None):
    """Convenience function to raise model training error"""
    raise ModelTrainingError(message, model_name=model_name)


def raise_configuration_error(message: str, section: str = None):
    """Convenience function to raise configuration error"""
    raise ConfigurationError(message, config_section=section)


def raise_dependency_error(package: str, install_cmd: str = None):
    """Convenience function to raise dependency error"""
    message = f"Required package '{package}' is not installed"
    if install_cmd:
        message += f". Install with: {install_cmd}"
    raise DependencyError(message, missing_package=package, install_command=install_cmd)


# Error handling decorator
def handle_pipeline_errors(func):
    """Decorator to handle and log pipeline errors"""
    import functools
    import logging
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        try:
            return func(*args, **kwargs)
        except PipelineError as e:
            logger.error(f"Pipeline error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise PipelineError(f"Unexpected error in {func.__name__}: {str(e)}")
    
    return wrapper
