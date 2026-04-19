#!/usr/bin/env python3

import sys
import os
import logging
import numpy as np

# Add current directory to path
sys.path.append('.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_complete_pipeline():
    """Test the complete model generation and classification pipeline"""
    print("=== Testing Complete Pipeline ===")
    
    try:
        # Step 1: Test model generation
        print("\n1. Testing model generation...")
        import generate_models
        
        # Generate all models
        generate_models.generate_all_models()
        print("   Models generated successfully!")
        
        # Step 2: Test model availability detection
        print("\n2. Testing model availability detection...")
        import classify
        
        available_models = classify.get_available_models()
        print(f"   Available models: {available_models}")
        
        if not available_models:
            print("   ERROR: No models available!")
            return False
        
        # Step 3: Test model loading
        print("\n3. Testing model loading...")
        for model_name in available_models:
            try:
                classify.load_model(model_name)
                print(f"   {model_name}: Loaded successfully")
            except Exception as e:
                print(f"   {model_name}: Failed to load - {e}")
                return False
        
        # Step 4: Test classification with sample data
        print("\n4. Testing ECG classification...")
        
        # Create sample ECG data (similar to test.tsv format)
        sample_data = np.random.randn(5, 96)  # 5 samples, 96 time points
        
        # Add some realistic ECG patterns
        for i in range(5):
            t = np.linspace(0, 2*np.pi, 96)
            sample_data[i] += 0.5 * np.sin(2*t) + 0.3 * np.sin(5*t)
        
        # Save sample data
        sample_file = 'test_sample_ecg.tsv'
        np.savetxt(sample_file, sample_data, delimiter='\t')
        
        # Test classification with each available model
        for model_name in available_models:
            try:
                classify.load_model(model_name)
                result = classify.classify_ecg(sample_file)
                print(f"   {model_name}: Classification successful")
                print(f"   Results: {result}")
            except Exception as e:
                print(f"   {model_name}: Classification failed - {e}")
                return False
        
        # Clean up
        os.remove(sample_file)
        
        print("\n=== Pipeline Test Complete ===")
        print("All tests passed! The pipeline is working correctly.")
        return True
        
    except Exception as e:
        print(f"\nERROR: Pipeline test failed - {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\nReady to start the Flask application!")
    else:
        print("\nPlease fix the issues before starting the application.")
