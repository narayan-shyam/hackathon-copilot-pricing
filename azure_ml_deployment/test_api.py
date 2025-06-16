"""
API Testing Script for Dynamic Pricing Model
This script tests the deployed Azure ML endpoint with various scenarios

Module 4: Testing Framework Implementation
"""

import requests
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Any
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicPricingAPITester:
    """
    Comprehensive testing suite for the Dynamic Pricing API
    """
    
    def __init__(self, scoring_uri: str, api_key: str):
        self.scoring_uri = scoring_uri
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.test_results = []
        
    def _make_request(self, data: Dict[str, Any], test_name: str) -> Dict[str, Any]:
        """Make API request and track results"""
        start_time = time.time()
        
        try:
            response = requests.post(
                self.scoring_uri,
                headers=self.headers,
                data=json.dumps(data),
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                self.test_results.append({
                    "test_name": test_name,
                    "status": "success",
                    "response_time": response_time,
                    "predicted_price": result.get("predicted_price"),
                    "input_data": data
                })
                logger.info(f"✅ {test_name}: Success (${result.get('predicted_price'):.2f}) in {response_time:.2f}s")
                return result
            else:
                self.test_results.append({
                    "test_name": test_name,
                    "status": "failed",
                    "response_time": response_time,
                    "error": response.text
                })
                logger.error(f"❌ {test_name}: Failed ({response.status_code}) - {response.text}")
                return {"error": response.text}
                
        except Exception as e:
            response_time = time.time() - start_time
            self.test_results.append({
                "test_name": test_name,
                "status": "error",
                "response_time": response_time,
                "error": str(e)
            })
            logger.error(f"❌ {test_name}: Exception - {str(e)}")
            return {"error": str(e)}
    
    def test_basic_prediction(self) -> Dict[str, Any]:
        """Test basic price prediction"""
        data = {
            "base_price": 100.0,
            "cost": 60.0,
            "competitor_price": 95.0,
            "demand": 150,
            "inventory_level": 500,
            "customer_engagement": 0.75,
            "market_demand_factor": 1.2,
            "seasonal_factor": 1.1,
            "price_to_cost_ratio": 1.67,
            "profit_margin": 0.4,
            "is_profitable": True,
            "price_vs_competitor": 1.05,
            "price_change": 2.0,
            "demand_change": 10,
            "price_elasticity": -0.5,
            "demand_trend_7d": 145.0,
            "price_volatility_7d": 5.2,
            "avg_price_30d": 98.5,
            "inventory_velocity": 0.3,
            "competitive_position": 1.05,
            "revenue_per_unit": 100.0,
            "profit_per_unit": 40.0,
            "day_of_week": 3,
            "month": 6,
            "quarter": 2,
            "is_weekend": 0,
            "is_month_end": 0,
            "category_avg_price": 102.0,
            "category_price_rank": 5,
            "price_vs_category_avg": 0.98,
            "demand_supply_ratio": 0.3,
            "profit_optimization_score": 12.0,
            "market_positioning_score": 1.27
        }
        
        return self._make_request(data, "Basic Prediction")
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report"""
        logger.info("🧪 Starting Comprehensive API Test Suite")
        logger.info("=" * 50)
        
        test_suite_start = time.time()
        
        # Run basic test
        logger.info("\n📋 Running Basic Test...")
        basic_result = self.test_basic_prediction()
        
        test_suite_time = time.time() - test_suite_start
        
        # Generate comprehensive report
        successful_tests = len([t for t in self.test_results if t["status"] == "success"])
        total_tests = len(self.test_results)
        
        comprehensive_report = {
            "test_summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
                "total_test_time": test_suite_time
            },
            "basic_test": basic_result,
            "detailed_results": self.test_results,
            "test_timestamp": datetime.now().isoformat()
        }
        
        # Log summary
        logger.info("\n🎯 TEST SUITE SUMMARY")
        logger.info("=" * 30)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Success Rate: {comprehensive_report['test_summary']['success_rate']:.1f}%")
        logger.info(f"Total Time: {test_suite_time:.2f}s")
        
        return comprehensive_report

def main():
    """Main testing function"""
    print("🧪 Dynamic Pricing API Testing Suite")
    print("=" * 40)
    
    # Load configuration from deployment info or environment
    scoring_uri = os.getenv("SCORING_URI")
    api_key = os.getenv("API_KEY")
    
    # Try to load from deployment info file
    if not scoring_uri or not api_key:
        try:
            with open("deployment_info.json", "r") as f:
                deployment_info = json.load(f)
                scoring_uri = deployment_info["endpoint_info"]["scoring_uri"]
                api_key = deployment_info["endpoint_info"]["primary_key"]
                print("✅ Loaded configuration from deployment_info.json")
        except FileNotFoundError:
            print("❌ deployment_info.json not found")
            print("💡 Please set SCORING_URI and API_KEY environment variables")
            return
    
    if not scoring_uri or not api_key:
        print("❌ Missing required configuration:")
        print("   - SCORING_URI: Azure ML endpoint URL")
        print("   - API_KEY: Azure ML endpoint key")
        return
    
    # Initialize tester
    tester = DynamicPricingAPITester(scoring_uri, api_key)
    
    # Run comprehensive test suite
    test_report = tester.run_comprehensive_test_suite()
    
    # Save test report
    report_filename = f"api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, "w") as f:
        json.dump(test_report, f, indent=2, default=str)
    
    print(f"\n📄 Test report saved to: {report_filename}")
    print("\n🎉 API Testing Complete!")
    print("🚀 Ready for Module 5: Monitoring & Logging Infrastructure!")

if __name__ == "__main__":
    main()
