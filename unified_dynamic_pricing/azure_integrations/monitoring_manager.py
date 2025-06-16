"""
Azure Monitoring Manager
Comprehensive monitoring and observability integration
"""

import logging
from typing import Dict, Any, Optional, List
import json
import time
from datetime import datetime, timedelta

# Azure Monitor imports with error handling
try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter, AzureMonitorMetricExporter
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    AZURE_MONITOR_AVAILABLE = True
except ImportError:
    AZURE_MONITOR_AVAILABLE = False

# Application Insights for custom telemetry
try:
    from applicationinsights import TelemetryClient
    from applicationinsights.logging import LoggingHandler
    APP_INSIGHTS_AVAILABLE = True
except ImportError:
    APP_INSIGHTS_AVAILABLE = False

logger = logging.getLogger(__name__)

class AzureMonitoringManager:
    """Azure Monitor integration for comprehensive observability"""
    
    def __init__(self, connection_string: str = None, instrumentation_key: str = None,
                 service_name: str = "dynamic-pricing-pipeline"):
        
        self.connection_string = connection_string
        self.instrumentation_key = instrumentation_key
        self.service_name = service_name
        self.telemetry_client = None
        self.tracer = None
        self.meter = None
        
        # Initialize monitoring
        self._initialize_azure_monitor()
        self._initialize_application_insights()
        self._setup_custom_metrics()
    
    def _initialize_azure_monitor(self):
        """Initialize Azure Monitor OpenTelemetry"""
        if not AZURE_MONITOR_AVAILABLE or not self.connection_string:
            logger.warning("Azure Monitor OpenTelemetry not available or connection string missing")
            return
        
        try:
            # Configure Azure Monitor
            configure_azure_monitor(
                connection_string=self.connection_string,
                service_name=self.service_name
            )
            
            # Get tracer and meter
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
            
            # Instrument common libraries
            RequestsInstrumentor().instrument()
            LoggingInstrumentor().instrument()
            
            logger.info("Azure Monitor OpenTelemetry configured")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Monitor: {e}")
    
    def _initialize_application_insights(self):
        """Initialize Application Insights client"""
        if not APP_INSIGHTS_AVAILABLE or not self.instrumentation_key:
            logger.warning("Application Insights not available or instrumentation key missing")
            return
        
        try:
            self.telemetry_client = TelemetryClient(self.instrumentation_key)
            
            # Add Application Insights logging handler
            ai_handler = LoggingHandler(self.instrumentation_key)
            ai_handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(ai_handler)
            
            logger.info("Application Insights configured")
            
        except Exception as e:
            logger.error(f"Failed to initialize Application Insights: {e}")
    
    def _setup_custom_metrics(self):
        """Setup custom metrics for pricing pipeline"""
        if not self.meter:
            return
        
        try:
            # Pipeline performance metrics
            self.pipeline_duration_histogram = self.meter.create_histogram(
                name="pipeline_duration_seconds",
                description="Time taken for pipeline execution",
                unit="s"
            )
            
            self.model_accuracy_gauge = self.meter.create_observable_gauge(
                name="model_accuracy_score",
                description="Model accuracy score (R2)",
                unit="1"
            )
            
            self.prediction_counter = self.meter.create_counter(
                name="predictions_total",
                description="Total number of predictions made",
                unit="1"
            )
            
            self.error_counter = self.meter.create_counter(
                name="errors_total",
                description="Total number of errors",
                unit="1"
            )
            
            logger.info("Custom metrics configured")
            
        except Exception as e:
            logger.error(f"Failed to setup custom metrics: {e}")
    
    def track_pipeline_execution(self, pipeline_name: str, duration_seconds: float,
                                success: bool, metrics: Dict[str, float] = None):
        """Track pipeline execution metrics"""
        try:
            # Record duration histogram
            if self.pipeline_duration_histogram:
                self.pipeline_duration_histogram.record(
                    duration_seconds,
                    attributes={
                        "pipeline.name": pipeline_name,
                        "pipeline.success": str(success)
                    }
                )
            
            # Track with Application Insights
            if self.telemetry_client:
                properties = {
                    'pipeline_name': pipeline_name,
                    'success': str(success),
                    'duration_seconds': duration_seconds
                }
                
                if metrics:
                    properties.update({f'metric_{k}': str(v) for k, v in metrics.items()})
                
                self.telemetry_client.track_event(
                    'PipelineExecution',
                    properties=properties,
                    measurements={'duration': duration_seconds}
                )
            
            logger.info(f"Tracked pipeline execution: {pipeline_name}")
            
        except Exception as e:
            logger.error(f"Failed to track pipeline execution: {e}")
    
    def track_model_performance(self, model_name: str, metrics: Dict[str, float],
                               dataset_size: int = None, training_time: float = None):
        """Track model performance metrics"""
        try:
            # Track with Application Insights
            if self.telemetry_client:
                properties = {
                    'model_name': model_name,
                    'dataset_size': str(dataset_size) if dataset_size else 'unknown'
                }
                
                measurements = {}
                if training_time:
                    measurements['training_time'] = training_time
                
                # Add all metrics as measurements
                measurements.update(metrics)
                
                self.telemetry_client.track_event(
                    'ModelPerformance',
                    properties=properties,
                    measurements=measurements
                )
            
            logger.info(f"Tracked model performance: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to track model performance: {e}")
    
    def track_error(self, error: Exception, operation: str = None, 
                   context: Dict[str, str] = None):
        """Track errors and exceptions"""
        try:
            # Increment error counter
            if self.error_counter:
                self.error_counter.add(
                    1,
                    attributes={
                        "error.type": type(error).__name__,
                        "operation": operation or "unknown"
                    }
                )
            
            # Track with Application Insights
            if self.telemetry_client:
                properties = {
                    'error_type': type(error).__name__,
                    'operation': operation or 'unknown',
                    'error_message': str(error)
                }
                
                if context:
                    properties.update(context)
                
                self.telemetry_client.track_exception(
                    type(error),
                    error,
                    error.__traceback__,
                    properties=properties
                )
            
            logger.error(f"Tracked error: {type(error).__name__} in {operation}")
            
        except Exception as e:
            logger.error(f"Failed to track error: {e}")
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get monitoring system health check"""
        health = {
            'azure_monitor_available': AZURE_MONITOR_AVAILABLE,
            'app_insights_available': APP_INSIGHTS_AVAILABLE,
            'connection_string_configured': bool(self.connection_string),
            'instrumentation_key_configured': bool(self.instrumentation_key),
            'telemetry_client_active': self.telemetry_client is not None,
            'tracer_active': self.tracer is not None,
            'meter_active': self.meter is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        return health


# Context manager for monitored operations
class MonitoredOperation:
    """Context manager for monitoring operations"""
    
    def __init__(self, monitoring_manager: AzureMonitoringManager, 
                 operation_name: str, operation_id: str = None):
        self.monitoring_manager = monitoring_manager
        self.operation_name = operation_name
        self.operation_id = operation_id
        self.start_time = None
        self.metrics = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        # Add duration to metrics
        self.metrics['duration_seconds'] = duration
        
        if exc_val:
            self.monitoring_manager.track_error(
                error=exc_val,
                operation=self.operation_name,
                context={'operation_id': self.operation_id} if self.operation_id else None
            )
        
        return False  # Don't suppress exceptions
    
    def add_metric(self, name: str, value: float):
        """Add custom metric to the operation"""
        self.metrics[name] = value
    
    def add_metrics(self, metrics: Dict[str, float]):
        """Add multiple metrics to the operation"""
        self.metrics.update(metrics)
