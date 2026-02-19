#!/usr/bin/env python3
"""
Quick test to validate that the grade handling fix works correctly
"""

import numpy as np
import sys
sys.path.append('/home/chris/moonboard')
from moonboard_data_prep import prepare_moonboard_data

def test_grade_handling():
    """Test that grades are properly handled and in valid range"""
    
    print("Testing grade handling fix...")
    
    # Prepare a small sample of data
    try:
        x_train, x_test, y_train, y_test, problem_df = prepare_moonboard_data(
            "data/problems MoonBoard 2016 .json",
            test_size=100, 
            random_state=42
        )
        
        print("\n=== VALIDATION RESULTS ===")
        
        # Check for invalid values
        print(f"Training data shape: {x_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        
        # Check for NaN values
        train_nan_count = np.sum(np.isnan(y_train))
        test_nan_count = np.sum(np.isnan(y_test))
        
        print(f"\nNaN values in training grades: {train_nan_count}")
        print(f"NaN values in test grades: {test_nan_count}")
        
        # Check grade ranges
        print(f"\nTraining grade range: {y_train.min():.0f} - {y_train.max():.0f}")
        print(f"Test grade range: {y_test.min():.0f} - {y_test.max():.0f}")
        
        # Check for zeros or negative values
        train_invalid = np.sum((y_train <= 0) | (y_train > 14))
        test_invalid = np.sum((y_test <= 0) | (y_test > 14))
        
        print(f"\nInvalid training grades (<=0 or >14): {train_invalid}")
        print(f"Invalid test grades (<=0 or >14): {test_invalid}")
        
        # Show grade distribution
        unique_train_grades = np.unique(y_train)
        unique_test_grades = np.unique(y_test)
        
        print(f"\nUnique training grades: {unique_train_grades}")
        print(f"Unique test grades: {unique_test_grades}")
        
        # Validation summary
        print("\n=== VALIDATION SUMMARY ===")
        all_good = True
        
        if train_nan_count > 0 or test_nan_count > 0:
            print("❌ FAIL: Found NaN values in grades")
            all_good = False
        else:
            print("✅ PASS: No NaN values found")
        
        if train_invalid > 0 or test_invalid > 0:
            print("❌ FAIL: Found invalid grade values")
            all_good = False
        else:
            print("✅ PASS: All grades in valid range [1-14]")
        
        if y_train.min() >= 1 and y_train.max() <= 14 and y_test.min() >= 1 and y_test.max() <= 14:
            print("✅ PASS: Grade ranges are correct")
        else:
            print("❌ FAIL: Grade ranges are outside expected bounds")
            all_good = False
        
        if all_good:
            print("\n🎉 ALL TESTS PASSED! The grade handling fix is working correctly.")
            print("   No more extreme predictions like -270 should occur.")
        else:
            print("\n⚠️  SOME TESTS FAILED! Check the issues above.")
        
        return all_good
        
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_grade_handling()
    sys.exit(0 if success else 1)