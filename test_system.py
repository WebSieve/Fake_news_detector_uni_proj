#!/usr/bin/env python3
"""
Comprehensive system test for the fake news detection project
"""

import sys
import traceback
from pathlib import Path

def test_configuration():
    """Test configuration module"""
    print("\n1️⃣ Testing Configuration...")
    try:
        from config import Config
        config = Config()
        
        # Check required attributes
        required_attrs = ['PROJECT_ROOT', 'DATA_DIR', 'MODELS_DIR', 'MODEL_SAVE_PATH', 
                         'MODEL_NAME', 'MAX_LENGTH', 'BATCH_SIZE', 'DEVICE']
        
        for attr in required_attrs:
            if not hasattr(config, attr):
                raise AttributeError(f"Missing required attribute: {attr}")
        
        print(f"   ✅ Model: {config.MODEL_NAME}")
        print(f"   ✅ Device: {config.DEVICE}")
        print(f"   ✅ Batch Size: {config.BATCH_SIZE}")
        print("✅ Configuration: PASSED")
        return True
    except Exception as e:
        print(f"❌ Configuration: FAILED - {e}")
        traceback.print_exc()
        return False

def test_utilities():
    """Test utilities module"""
    print("\n2️⃣ Testing Utilities...")
    try:
        from utils import setup_logging, clean_text, save_json, load_json, create_dir
        
        # Test logging
        logger = setup_logging('test')
        
        # Test text cleaning
        test_text = "This is a test article with URL https://example.com and email test@test.com"
        cleaned = clean_text(test_text)
        if not isinstance(cleaned, str):
            raise TypeError("clean_text should return string")
        
        # Test directory creation
        create_dir('test_temp_dir')
        
        # Test JSON operations
        test_data = {"test": "data", "number": 42}
        save_json(test_data, 'test_temp_dir/test.json')
        loaded_data = load_json('test_temp_dir/test.json')
        
        if loaded_data != test_data:
            raise ValueError("JSON save/load failed")
        
        print("   ✅ Logging setup works")
        print("   ✅ Text cleaning works")
        print("   ✅ JSON operations work")
        print("✅ Utilities: PASSED")
        return True
    except Exception as e:
        print(f"❌ Utilities: FAILED - {e}")
        traceback.print_exc()
        return False

def test_simple_inference():
    """Test simple inference module"""
    print("\n3️⃣ Testing Simple Inference...")
    try:
        from fake_news_inference import SimplePredictor
        
        # Initialize predictor
        predictor = SimplePredictor()
        
        # Test prediction
        test_articles = [
            "Scientists discover new medical breakthrough in cancer research.",
            "SHOCKING: This one weird trick will cure everything!",
            "Local weather forecast predicts rain for tomorrow."
        ]
        
        for i, article in enumerate(test_articles):
            result = predictor.predict(article)
            
            # Validate result format
            required_keys = ['prediction', 'label', 'confidence', 'fake_probability']
            for key in required_keys:
                if key not in result:
                    raise KeyError(f"Missing key in result: {key}")
            
            if result['label'] not in ['REAL', 'FAKE']:
                raise ValueError(f"Invalid label: {result['label']}")
            
            if not 0 <= result['confidence'] <= 1:
                raise ValueError(f"Invalid confidence: {result['confidence']}")
            
            print(f"   ✅ Test {i+1}: {result['label']} ({result['confidence']:.2%})")
        
        print("✅ Simple Inference: PASSED")
        return True
    except Exception as e:
        print(f"❌ Simple Inference: FAILED - {e}")
        traceback.print_exc()
        return False

def test_data_preparation():
    """Test data preparation module"""
    print("\n4️⃣ Testing Data Preparation...")
    try:
        from data_preparation import DatasetLoader
        
        # Initialize loader
        loader = DatasetLoader()
        
        # Create sample dataset
        sample_data = loader.create_sample_dataset(n_samples=50)
        
        # Validate dataset
        if len(sample_data) != 50:
            raise ValueError(f"Expected 50 samples, got {len(sample_data)}")
        
        required_columns = ['text', 'label']
        for col in required_columns:
            if col not in sample_data.columns:
                raise ValueError(f"Missing column: {col}")
        
        # Check label distribution
        label_counts = sample_data['label'].value_counts()
        if len(label_counts) != 2:
            raise ValueError("Dataset should have exactly 2 labels")
        
        print(f"   ✅ Dataset size: {len(sample_data)}")
        print(f"   ✅ Columns: {list(sample_data.columns)}")
        print(f"   ✅ Label distribution: {dict(label_counts)}")
        print("✅ Data Preparation: PASSED")
        return True
    except Exception as e:
        print(f"❌ Data Preparation: FAILED - {e}")
        traceback.print_exc()
        return False

def test_imports():
    """Test all critical imports"""
    print("\n5️⃣ Testing All Imports...")
    try:
        import config
        import utils
        import data_preparation
        import fake_news_inference
        import model_training
        import evaluation
        import web_application
        import project_pipeline
        
        print("   ✅ All modules import successfully")
        print("✅ Import Test: PASSED")
        return True
    except Exception as e:
        print(f"❌ Import Test: FAILED - {e}")
        traceback.print_exc()
        return False

def test_file_integrity():
    """Test file integrity and structure"""
    print("\n6️⃣ Testing File Integrity...")
    try:
        required_files = [
            'config.py', 'utils.py', 'data_preparation.py', 'fake_news_inference.py',
            'model_training.py', 'evaluation.py', 'web_application.py', 'project_pipeline.py',
            'requirements.txt', 'README.md', 'DEPLOYMENT.md',
            'Dockerfile', 'docker-compose.yml'
        ]
        
        project_root = Path('.')
        missing_files = []
        
        for file in required_files:
            if not (project_root / file).exists():
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")
        
        # Check directories
        required_dirs = ['data', 'models', 'logs']
        for dir_name in required_dirs:
            if not (project_root / dir_name).exists():
                print(f"   ⚠️ Creating missing directory: {dir_name}")
                (project_root / dir_name).mkdir(exist_ok=True)
        
        print(f"   ✅ All {len(required_files)} required files present")
        print("   ✅ Directory structure validated")
        print("✅ File Integrity: PASSED")
        return True
    except Exception as e:
        print(f"❌ File Integrity: FAILED - {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive system test"""
    print("🔍 COMPREHENSIVE FAKE NEWS DETECTION SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        test_configuration,
        test_utilities,
        test_simple_inference,
        test_data_preparation,
        test_imports,
        test_file_integrity
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment.")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
