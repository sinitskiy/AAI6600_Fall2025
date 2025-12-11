#!/usr/bin/env python3
"""
Test Mental Health Classifier
Comprehensive testing of intent classification
"""

import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from core.mental_health_classifier import MentalHealthClassifier


def run_classifier_tests():
    """Run comprehensive classifier tests"""
    
    print("=" * 70)
    print("MENTAL HEALTH CLASSIFIER - COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Initialize classifier
    print("\n[Initialization]")
    classifier = MentalHealthClassifier()
    print("✓ Classifier loaded\n")
    
    # Test cases with expected intents
    test_cases = [
        {
            'input': "I want to kill myself",
            'expected_intent': 'suicide',
            'min_confidence': 0.8
        },
        {
            'input': "I'm feeling really anxious about my exams",
            'expected_intent': 'anxious',
            'min_confidence': 0.6
        },
        {
            'input': "I feel so sad and depressed",
            'expected_intent': 'sad',
            'min_confidence': 0.6
        },
        {
            'input': "I'm so stressed and overwhelmed with work",
            'expected_intent': 'stressed',
            'min_confidence': 0.6
        },
        {
            'input': "I need help finding a therapist",
            'expected_intent': 'help',
            'min_confidence': 0.5
        },
        {
            'input': "Hello, how are you?",
            'expected_intent': 'greeting',
            'min_confidence': 0.5
        },
        {
            'input': "I'm failing all my classes and can't handle school",
            'expected_intent': 'academic-pressure',
            'min_confidence': 0.5
        },
        {
            'input': "I feel so alone and have no friends",
            'expected_intent': 'loneliness',
            'min_confidence': 0.5
        }
    ]
    
    passed = 0
    failed = 0
    
    print("[Test Cases]")
    print("-" * 70)
    
    for i, test in enumerate(test_cases, 1):
        user_input = test['input']
        expected = test['expected_intent']
        min_conf = test['min_confidence']
        
        result = classifier.classify_with_response(user_input)
        intent = result['intent']
        confidence = result['confidence']
        is_crisis = result['is_crisis']
        
        # Check if intent matches
        intent_match = intent == expected
        confidence_ok = confidence >= min_conf
        
        if intent_match and confidence_ok:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        print(f"\nTest {i}: {status}")
        print(f"  Input: {user_input}")
        print(f"  Expected: {expected}")
        print(f"  Got: {intent} (confidence: {confidence:.2%})")
        print(f"  Crisis: {is_crisis}")
        
        if not intent_match:
            print(f"  ⚠️  Intent mismatch!")
        if not confidence_ok:
            print(f"  ⚠️  Low confidence (expected >{min_conf:.0%})")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {failed}/{len(test_cases)}")
    print(f"Success Rate: {passed/len(test_cases)*100:.1f}%")
    
    if passed == len(test_cases):
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️  Some tests failed. Review results above.")
    
    print("=" * 70)
    
    return passed == len(test_cases)


if __name__ == "__main__":
    try:
        success = run_classifier_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)