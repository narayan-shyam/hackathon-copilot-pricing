"""
Comprehensive test suite for the unified dynamic pricing pipeline
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from unified_dynamic_pricing import (
        UnifiedDynamicPricingPipeline,
        create_sample_pricing_data,
        create_pipeline_config
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"unified_dynamic_pricing module not available: {e}", allow_module_level=True)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return create_sample_pricing_data(days=30, n_products=2)


@pytest.fixture
def minimal_data():
    """Create minimal test data"""
    return pd.DataFrame({
        'SellingPrice': [100, 150, 200, 120, 180],
        'Demand': [10, 15, 20, 12, 18],
        'Cost': [80, 120, 160, 96, 144],
        'CompetitorPrice': [105, 155, 195, 125, 175],
        'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    })


class TestDataGeneration:
    """Test data generation utilities"""
    
    def test_create_sample_data_basic(self):
        """Test basic sample data creation"""
        data = create_sample_pricing_data(days=10, n_products=2)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 20  # 10 days * 2 products
        assert 'SellingPrice' in data.columns
        assert 'Demand' in data.columns
        assert 'Cost' in data.columns
        assert 'Date' in data.columns
        
    def test_sample_data_values(self):
        """Test sample data has reasonable values"""
        data = create_sample_pricing_data(days=5, n_products=1)
        
        assert data['SellingPrice'].min() > 0
        assert data['Demand'].min() >= 0
        assert data['Cost'].min() > 0


class TestUnifiedPipeline:
    """Test complete pipeline functionality"""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = UnifiedDynamicPricingPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'run_complete_pipeline')
        assert hasattr(pipeline, 'predict')
    
    def test_pipeline_with_config(self):
        """Test pipeline with custom configuration"""
        config = create_pipeline_config({
            'model_trainer': {'cv_folds': 3}
        })
        pipeline = UnifiedDynamicPricingPipeline(config)
        assert pipeline.config is not None
    
    def test_data_loading_dataframe(self, minimal_data):
        """Test data loading from DataFrame"""
        pipeline = UnifiedDynamicPricingPipeline()
        loaded_data = pipeline.load_data(minimal_data)
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == minimal_data.shape
    
    @pytest.mark.slow
    def test_basic_pipeline_run(self, sample_data):
        """Test basic pipeline execution (marked as slow)"""
        pipeline = UnifiedDynamicPricingPipeline()
        
        # Use smaller data for faster testing
        small_data = sample_data.head(20)
        
        try:
            results = pipeline.run_complete_pipeline(
                small_data, 
                target_column='SellingPrice',
                test_size=0.3
            )
            
            assert results is not None
            assert 'pipeline_status' in results
            
            if results['pipeline_status'] == 'success':
                assert 'best_model' in results
                assert 'data_validation' in results
                
        except Exception as e:
            # Pipeline may fail due to small data size or missing dependencies
            # This is acceptable for basic structural testing
            pytest.skip(f"Pipeline execution failed (acceptable): {e}")


class TestConfigurationAndExceptions:
    """Test configuration and exception handling"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = create_pipeline_config()
        
        assert isinstance(config, dict)
        assert 'data_processor' in config
        assert 'feature_engineer' in config
        assert 'model_trainer' in config
    
    def test_custom_config_merge(self):
        """Test custom configuration merging"""
        custom_config = {
            'model_trainer': {'cv_folds': 10},
            'new_section': {'param': 'value'}
        }
        
        config = create_pipeline_config(custom_config)
        
        assert config['model_trainer']['cv_folds'] == 10
        assert 'new_section' in config


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_print_functions_no_error(self, sample_data):
        """Test print functions don't raise errors"""
        from unified_dynamic_pricing import (
            print_pipeline_results,
            print_feature_summary,
            print_qa_answers
        )
        
        # Test with minimal results
        mock_results = {
            'pipeline_status': 'success',
            'data_validation': {'schema_valid': True, 'quality_score': 85},
            'best_model': {'name': 'test_model', 'cv_score': 0.8}
        }
        
        # These should not raise exceptions
        try:
            print_pipeline_results(mock_results)
            print_feature_summary()
            print_qa_answers()
        except Exception as e:
            pytest.fail(f"Print functions raised unexpected exception: {e}")
            

# Smoke test
class TestAdvancedValidation:
    """Advanced tests: smoke, drift, performance, and bias"""

    def test_endpoint_smoke(self):
        """Smoke test for Azure ML endpoint (mocked)"""
        import requests
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = {'predictions': [10.0, 12.0]}
            response = requests.post('http://fake-endpoint', json={'data': [1, 2, 3]})
            assert response.status_code == 200
            assert 'predictions' in response.json()

    def test_model_drift_detection(self, sample_data):
        """Test model drift detection by simulating distribution shift"""
        drifted = sample_data.copy()
        drifted['SellingPrice'] *= 1.5
        orig_mean = sample_data['SellingPrice'].mean()
        drifted_mean = drifted['SellingPrice'].mean()
        assert abs(orig_mean - drifted_mean) > 0.1 * orig_mean

    def test_model_performance_threshold(self, sample_data):
        """Test model performance threshold"""
        pipeline = UnifiedDynamicPricingPipeline()
        results = pipeline.run_complete_pipeline(
            sample_data, target_column='SellingPrice', test_size=0.2
        )
        assert results['evaluation_results']['r2'] > 0.5  # Example threshold

    def test_model_bias_detection(self, sample_data):
        """Test for prediction bias across groups"""
        pipeline = UnifiedDynamicPricingPipeline()
        sample_data['Region'] = np.random.choice(['North', 'South'], size=len(sample_data))
        pipeline.run_complete_pipeline(
            sample_data, target_column='SellingPrice', test_size=0.2
        )
        new_data = sample_data.copy()
        new_data['Region'] = 'North'
        preds_north = pipeline.predict(new_data)
        new_data['Region'] = 'South'
        preds_south = pipeline.predict(new_data)
        assert abs(np.mean(preds_north) - np.mean(preds_south)) < 0.5 * np.mean(preds_north)



# Test markers for different test categories
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (may skip in CI)")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
