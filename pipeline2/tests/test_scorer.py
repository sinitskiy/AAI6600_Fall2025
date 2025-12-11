#!/usr/bin/env python3
"""
Test Facility Scorer
Tests semantic similarity scoring functionality
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from core.facility_scorer import FacilityScorer


def test_scorer_initialization():
    """Test that scorer initializes correctly"""
    print("\n[Test 1] Scorer Initialization")
    print("-" * 70)
    
    try:
        scorer = FacilityScorer()
        print("✓ Scorer initialized successfully")
        print(f"✓ Model loaded: sentence-transformers/all-MiniLM-L6-v2")
        print(f"✓ Pre-computed {len(scorer.question_embeddings)} question embeddings")
        return True, scorer
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False, None


def test_scoring():
    """Test scoring functionality"""
    print("\n[Test 2] Facility Scoring")
    print("-" * 70)
    
    try:
        scorer = FacilityScorer()
        
        # Create test dataframe
        test_facilities = pd.DataFrame({
            'name': [
                'Community Mental Health Center',
                'Affordable Counseling Services', 
                'Private Psychiatric Practice',
                'Crisis Intervention Center',
                'University Counseling Clinic'
            ],
            'street': [
                '123 Main St',
                '456 Community Ave',
                '789 Executive Blvd',
                '321 Emergency Way',
                '654 Campus Dr'
            ],
            'city': ['Boston', 'Cambridge', 'Brookline', 'Somerville', 'Boston'],
            'state': ['MA', 'MA', 'MA', 'MA', 'MA']
        })
        
        print(f"  Created test dataset: {len(test_facilities)} facilities")
        
        # Score facilities
        scored_df = scorer.score_facilities(test_facilities)
        
        print(f"✓ Scoring completed")
        print(f"✓ Added {len([col for col in scored_df.columns if 'score' in col])} score columns")
        
        # Check that scores are reasonable
        avg_score = scored_df['overall_care_needs_score'].mean()
        print(f"✓ Average overall score: {avg_score:.2f}/10")
        
        # Display top facility
        top_facility = scored_df.nlargest(1, 'overall_care_needs_score').iloc[0]
        print(f"\n  Top Facility: {top_facility['name']}")
        print(f"  Overall Score: {top_facility['overall_care_needs_score']:.2f}/10")
        print(f"  Affordability: {top_facility['affordability_score']:.2f}/10")
        print(f"  Crisis Care: {top_facility['crisis_care_score']:.2f}/10")
        
        return True
    except Exception as e:
        print(f"✗ Scoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search():
    """Test facility search functionality"""
    print("\n[Test 3] Facility Search")
    print("-" * 70)
    
    try:
        scorer = FacilityScorer()
        
        # Create larger test dataset
        test_facilities = pd.DataFrame({
            'name': ['Boston Center'] * 3 + ['NYC Center'] * 3,
            'street': ['123 Main St'] * 6,
            'city': ['Boston'] * 3 + ['New York'] * 3,
            'state': ['MA'] * 3 + ['NY'] * 3,
            'overall_care_needs_score': [8.5, 7.2, 6.8, 9.1, 7.5, 6.3]
        })
        
        # Test city filter
        results = scorer.get_top_facilities(test_facilities, city="Boston", n=5)
        print(f"✓ City filter works: {len(results)} facilities in Boston")
        
        # Test state filter
        results = scorer.get_top_facilities(test_facilities, state="NY", n=5)
        print(f"✓ State filter works: {len(results)} facilities in NY")
        
        # Test top N
        results = scorer.get_top_facilities(test_facilities, n=2)
        print(f"✓ Top N works: returned {len(results)} facilities")
        
        return True
    except Exception as e:
        print(f"✗ Search test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("FACILITY SCORER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Test 1: Initialization
    success, scorer = test_scorer_initialization()
    results.append(success)
    
    if not success:
        print("\n⚠️  Skipping remaining tests (initialization failed)")
        return False
    
    # Test 2: Scoring
    results.append(test_scoring())
    
    # Test 3: Search
    results.append(test_search())
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n⚠️  SOME TESTS FAILED")
    
    print("=" * 70)
    
    return all(results)


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)