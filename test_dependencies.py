#!/usr/bin/env python3
"""
Test script to verify all dependencies are correctly installed
and the basic functionality of the semiconductor analysis system is working.
"""

import os
import sys
import numpy as np

print("=== Testing Semiconductor Analysis System Dependencies ===")

# Test matplotlib
print("\nTesting matplotlib...", end="")
try:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.close()
    print(" ✓ OK")
except ImportError:
    print(" ✗ FAILED - matplotlib not installed")
    print("   Install with: pip install matplotlib")
except Exception as e:
    print(f" ✗ FAILED - {str(e)}")

# Test semantic search dependencies
print("\nTesting semantic search dependencies...")
try:
    print("  - scipy...", end="")
    import scipy.spatial
    print(" ✓ OK")
    
    print("  - scikit-learn...", end="")
    from sklearn.feature_extraction.text import TfidfVectorizer
    print(" ✓ OK")
    
    # Test actual functionality
    print("  - TF-IDF with small corpus...", end="")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(["silicon semiconductor", "gallium arsenide properties", "semiconductor bandgap"])
    print(" ✓ OK")
except ImportError as e:
    print(f" ✗ FAILED - {str(e)}")
    print("   Install with: pip install scipy scikit-learn")
except Exception as e:
    print(f" ✗ FAILED - {str(e)}")

# Test core modules
print("\nTesting core module imports...")

try:
    print("  - Importing evaluate_models module...", end="")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from evaluate_models import ResponseCache, APIRateManager, SemanticSearchEngine
    print(" ✓ OK")
    
    # Test instantiation
    print("  - Initializing API Rate Manager...", end="")
    api_manager = APIRateManager()
    print(" ✓ OK")
    
    print("  - Initializing Response Cache...", end="")
    cache = ResponseCache("test_cache")
    print(" ✓ OK")
    
    print("  - Initializing Semantic Search Engine...", end="")
    search_engine = SemanticSearchEngine()
    print(" ✓ OK")
    
except ImportError as e:
    print(f" ✗ FAILED - {str(e)}")
except Exception as e:
    print(f" ✗ FAILED - {str(e)}")

print("\n=== All dependency tests completed ===") 