#!/usr/bin/env python3
"""
Pipeline 2 - Smoke Test
Quick validation that all components load successfully
"""

import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

def test_imports():
    """Test that all modules import successfully"""
    print("=" * 70)
    print("PIPELINE 2 - SMOKE TEST")
    print("=" * 70)
    print("\n[Test 1] Testing imports...")
    
    try:
        from core.mental_health_classifier import MentalHealthClassifier
        print("✓ MentalHealthClassifier imported")
    except Exception as e:
        print(f"✗ MentalHealthClassifier failed: {e}")
        return False
    
    try:
        from core.facility_scorer import FacilityScorer
        print("✓ FacilityScorer imported")
    except Exception as e:
        print(f"✗ FacilityScorer failed: {e}")
        return False
    
    try:
        from core.anti_hallucination import AntiHallucinationValidator, NegativeEvidenceDetector
        print("✓ AntiHallucinationValidator imported")
    except Exception as e:
        print(f"✗ AntiHallucinationValidator failed: {e}")
        return False
    
    return True


def test_classifier():
    """Test mental health classifier"""
    print("\n[Test 2] Testing Mental Health Classifier...")
    
    try:
        from core.mental_health_classifier import MentalHealthClassifier
        
        classifier = MentalHealthClassifier()
        print("✓ Classifier initialized")
        
        # Test classification
        test_input = "I'm feeling really anxious"
        intent, confidence = classifier.predict(test_input)
        
        print(f"✓ Classification works: '{test_input}' → {intent} ({confidence:.2%})")
        
        return True
    except Exception as e:
        print(f"✗ Classifier test failed: {e}")
        return False


def test_scorer():
    """Test facility scorer"""
    print("\n[Test 3] Testing Facility Scorer...")
    
    try:
        from core.facility_scorer import FacilityScorer
        import pandas as pd
        
        scorer = FacilityScorer()
        print("✓ Scorer initialized (model loaded)")
        
        # Test with small dataframe
        test_df = pd.DataFrame({
            'name': ['Community Mental Health Center'],
            'street': ['123 Main St'],
            'city': ['Boston'],
            'state': ['MA']
        })
        
        scored_df = scorer.score_facilities(test_df)
        print(f"✓ Scoring works: {scored_df['overall_care_needs_score'].iloc[0]:.2f}/10")
        
        return True
    except Exception as e:
        print(f"✗ Scorer test failed: {e}")
        return False


def test_validator():
    """Test anti-hallucination validator"""
    print("\n[Test 4] Testing Anti-Hallucination Validator...")
    
    try:
        from core.anti_hallucination import AntiHallucinationValidator
        
        validator = AntiHallucinationValidator()
        print("✓ Validator initialized")
        
        # Test validation
        test_facility = {
            'name': 'Test Facility',
            'score': 8.5,
            'description': 'Affordable mental health services'
        }
        
        validated = validator.validate_facility(test_facility)
        print(f"✓ Validation works: reliability score = {validated['validation']['reliability_score']:.1f}/100")
        
        return True
    except Exception as e:
        print(f"✗ Validator test failed: {e}")
        return False


def main():
    """Run all smoke tests"""
    results = []
    
    results.append(test_imports())
    results.append(test_classifier())
    results.append(test_scorer())
    results.append(test_validator())
    
    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("\n🎉 Pipeline 2 is ready to use!")
        print("\nNext steps:")
        print("  python run_gui.py           # Launch GUI")
        print("  python main.py --help       # See CLI options")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total})")
        print("\nPlease fix the errors above before proceeding.")
        return False
    
    print("=" * 70)
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)