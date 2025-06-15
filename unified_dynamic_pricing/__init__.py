"""
Unified Dynamic Pricing Pipeline
Consolidated Module 1 and Module 2 with optimized, deduplicated code

This unified pipeline combines all functionality from both modules:
- Comprehensive data processing and validation
- Advanced feature engineering for pricing optimization  
- Multiple ML algorithms with hyperparameter optimization
- MLflow integration and experiment tracking
- Business-specific pricing metrics and evaluation
- Time series analysis and seasonal features
"""

from .pipeline.dynamic_pricing_pipeline import UnifiedDynamicPricingPipeline
from .pipeline.data_processor import UnifiedDataProcessor
from .pipeline.feature_engineer import UnifiedFeatureEngineer
from .pipeline.model_trainer import UnifiedModelTrainer
from .utils import (
    create_sample_pricing_data,
    print_pipeline_results,
    print_feature_summary,
    print_qa_answers,
    create_pipeline_config
)

__version__ = "1.0.0"
__author__ = "Dynamic Pricing Team"

__all__ = [
    # Main Pipeline
    'UnifiedDynamicPricingPipeline',
    
    # Pipeline Components
    'UnifiedDataProcessor',
    'UnifiedFeatureEngineer', 
    'UnifiedModelTrainer',
    
    # Utility Functions
    'create_sample_pricing_data',
    'print_pipeline_results',
    'print_feature_summary',
    'print_qa_answers',
    'create_pipeline_config'
]


def quick_start_demo():
    """Quick demonstration of the unified pipeline"""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 UNIFIED DYNAMIC PRICING PIPELINE - QUICK START DEMO")
    print("=" * 70)
    
    # Create sample data
    print("\n📊 Creating sample pricing data...")
    sample_data = create_sample_pricing_data(n_days=180, n_products=3)
    print(f"Generated {len(sample_data)} records for {sample_data['Product'].nunique()} products")
    
    # Initialize pipeline with optimized config
    print("\n⚙️ Initializing unified pipeline...")
    config = create_pipeline_config()
    pipeline = UnifiedDynamicPricingPipeline(config)
    
    # Run complete pipeline
    print("\n🔄 Running complete pipeline...")
    results = pipeline.run_complete_pipeline(
        data_source=sample_data,
        target_column='SellingPrice',
        test_size=0.2
    )
    
    # Print results
    print_pipeline_results(results)
    
    # Print Q&A answers
    print_qa_answers()
    
    # Print feature summary
    print_feature_summary()
    
    return pipeline, results


if __name__ == "__main__":
    # Run demo if script is executed directly
    pipeline, results = quick_start_demo()
