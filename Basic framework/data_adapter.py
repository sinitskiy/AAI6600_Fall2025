#!/usr/bin/env python3
"""
Data Adapter - Translation Layer for Multiple Input Sources

This module standardizes data from different sources (LLM, mock classifier,
TXT files) into a consistent format that group2_router.py expects.

Purpose:
- Decouple input sources from routing logic
- Validate and normalize data
- Make it easy to swap between mock/real LLM/TXT sources
- Prevent errors from malformed data

Author: Subgroup B (Michael & Radhika)
Date: November 2025
"""

import re
from typing import Dict, Optional, Any


# =====================================================
# Standard Output Format (What Router Expects)
# =====================================================

REQUIRED_FIELDS = ['category', 'confidence', 'user_input']

STANDARD_FORMAT = {
    'category': str,      # e.g., 'Mental health', 'Crisis counseling'
    'confidence': float,  # 0.0 to 1.0 (e.g., 0.85 for 85%)
    'user_input': str     # Original user text or summary
}


# =====================================================
# Category Normalization
# =====================================================

# Mapping of common variations to standard category names
CATEGORY_ALIASES = {
    # Mental health variations
    'mental health': 'Mental health',
    'mental_health': 'Mental health',
    'mentalhealth': 'Mental health',
    'mental health support': 'Mental health support',
    
    # Crisis variations
    'crisis': 'Crisis counseling',
    'crisis counseling': 'Crisis counseling',
    'crisis_counseling': 'Crisis counseling',
    'emergency': 'Crisis counseling',
    'urgent': 'Crisis counseling',
    
    # Counseling variations
    'counseling': 'Counseling',
    'therapy': 'Counseling',
    'therapist': 'Counseling',
    
    # Add more mappings as needed
}


def normalize_category(category: str) -> str:
    """
    Normalize category names to standard format
    
    Handles:
    - Case variations (MENTAL HEALTH → Mental health)
    - Underscores vs spaces (mental_health → Mental health)
    - Common aliases (therapy → Counseling)
    
    Args:
        category: Raw category string
    
    Returns:
        str: Normalized category name
    """
    
    if not category or not isinstance(category, str):
        return 'Unknown'
    
    # Clean up the string
    cleaned = category.strip().lower()
    
    # Check aliases first
    if cleaned in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[cleaned]
    
    # Default: capitalize first letter of each word
    return category.strip().title()


def normalize_confidence(confidence: Any) -> float:
    """
    Normalize confidence values to 0.0-1.0 range
    
    Handles:
    - Percentages (85 → 0.85)
    - Already normalized (0.85 → 0.85)
    - String numbers ("85" → 0.85)
    - Out of range values (clip to 0-1)
    
    Args:
        confidence: Raw confidence value (int, float, or str)
    
    Returns:
        float: Normalized confidence between 0.0 and 1.0
    """
    
    try:
        # Convert to float
        if isinstance(confidence, str):
            confidence = float(confidence)
        
        conf = float(confidence)
        
        # If it looks like a percentage (> 1), convert to decimal
        if conf > 1.0:
            conf = conf / 100.0
        
        # Clip to valid range
        conf = max(0.0, min(1.0, conf))
        
        return conf
        
    except (ValueError, TypeError):
        # Default to medium confidence if we can't parse
        return 0.5


# =====================================================
# Validation
# =====================================================

def validate_classification(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate that classification data has all required fields
    
    Args:
        data: Classification dictionary
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    
    if not isinstance(data, dict):
        return False, "Data must be a dictionary"
    
    # Check required fields
    missing_fields = []
    for field in REQUIRED_FIELDS:
        if field not in data:
            missing_fields.append(field)
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Check types
    if not isinstance(data['category'], str) or not data['category'].strip():
        return False, "Category must be a non-empty string"
    
    # Confidence should be convertible to float
    try:
        conf = float(data['confidence'])
        if conf < 0 or conf > 100:  # Allow both formats for validation
            return False, "Confidence must be between 0-1 or 0-100"
    except (ValueError, TypeError):
        return False, "Confidence must be a number"
    
    # user_input should be string (can be empty)
    if not isinstance(data['user_input'], str):
        return False, "user_input must be a string"
    
    return True, None


# =====================================================
# Adapter Functions
# =====================================================

def adapt_mock_output(mock_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt mock classifier output to standard format
    
    Mock classifier should already be in correct format, but this
    provides normalization and validation.
    
    Args:
        mock_data: Output from mock_classify_conversation()
    
    Returns:
        dict: Standardized classification data
    
    Example:
        Input: {'category': 'mental health', 'confidence': 0.85, 'user_input': '...'}
        Output: {'category': 'Mental health', 'confidence': 0.85, 'user_input': '...'}
    """
    
    # Validate first
    is_valid, error = validate_classification(mock_data)
    if not is_valid:
        raise ValueError(f"Invalid mock data: {error}")
    
    return {
        'category': normalize_category(mock_data['category']),
        'confidence': normalize_confidence(mock_data['confidence']),
        'user_input': str(mock_data['user_input'])
    }


def adapt_llm_output(llm_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt Subgroup A's LLM output to standard format
    
    This function handles various possible formats from the LLM and
    translates them to what group2_router expects.
    
    Expected LLM output formats (flexible):
    - {'category': '...', 'confidence': ..., 'symptoms': '...'}
    - {'intent': '...', 'certainty': ..., 'user_text': '...'}
    - {'classification': '...', 'score': ..., 'input': '...'}
    
    Args:
        llm_data: Output from Subgroup A's LLM classifier
    
    Returns:
        dict: Standardized classification data
    
    Example:
        Input: {'intent': 'depression', 'certainty': 85, 'symptoms': 'feeling down'}
        Output: {'category': 'Depression', 'confidence': 0.85, 'user_input': 'feeling down'}
    """
    
    if not isinstance(llm_data, dict):
        raise ValueError("LLM output must be a dictionary")
    
    # Extract category (try multiple possible field names)
    category = None
    for key in ['category', 'classification', 'intent', 'class', 'label']:
        if key in llm_data:
            category = llm_data[key]
            break
    
    if not category:
        raise ValueError("LLM output missing category/classification/intent field")
    
    # Extract confidence (try multiple possible field names)
    confidence = None
    for key in ['confidence', 'certainty', 'score', 'probability', 'conf']:
        if key in llm_data:
            confidence = llm_data[key]
            break
    
    if confidence is None:
        # Default to medium confidence if not provided
        confidence = 0.7
        print(f"Warning: No confidence field found in LLM output, using default {confidence}")
    
    # Extract user input (try multiple possible field names)
    user_input = None
    for key in ['user_input', 'input', 'text', 'symptoms', 'description', 'query', 'user_text']:
        if key in llm_data:
            user_input = llm_data[key]
            break
    
    if not user_input:
        user_input = "LLM classification result"
    
    # Build standardized output
    standardized = {
        'category': normalize_category(category),
        'confidence': normalize_confidence(confidence),
        'user_input': str(user_input)
    }
    
    # Preserve optional fields returned by LLM (if present)
    if 'symptoms' in llm_data:
        standardized['symptoms'] = str(llm_data.get('symptoms'))
    # Location may be nested dict or separate fields
    if 'location' in llm_data and isinstance(llm_data['location'], dict):
        standardized['location'] = llm_data['location']
    else:
        # Try to pick up city/state fields if present
        city = llm_data.get('city') or llm_data.get('city_raw')
        state = llm_data.get('state') or llm_data.get('state_raw')
        if city or state:
            standardized['location'] = {'city': city or '', 'state': state or ''}

    if 'insurance' in llm_data and isinstance(llm_data['insurance'], dict):
        standardized['insurance'] = llm_data['insurance']
    else:
        # try common insurance fields
        if any(k in llm_data for k in ('has_insurance', 'provider')):
            standardized['insurance'] = {
                'has_insurance': bool(llm_data.get('has_insurance', False)),
                'provider': llm_data.get('provider', '')
            }
    
    # Validate before returning
    is_valid, error = validate_classification(standardized)
    if not is_valid:
        raise ValueError(f"Adapted LLM data is invalid: {error}")
    
    return standardized


def adapt_txt_output(txt_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt Group 2's TXT file output to standard format
    
    Handles the format from test.txt files generated by test.py
    
    Args:
        txt_data: Parsed data from test.txt (from router_txt_input.py)
    
    Returns:
        dict: Standardized classification data
    
    Example:
        Input: {'category': 'Mental health', 'confidence': 0.95, 'user_input': '...', 'scenario_name': '...'}
        Output: {'category': 'Mental health', 'confidence': 0.95, 'user_input': '...'}
    """
    
    if not isinstance(txt_data, dict):
        raise ValueError("TXT data must be a dictionary")
    
    # TXT format should already be close to standard, just normalize
    required_fields = ['category', 'confidence']
    for field in required_fields:
        if field not in txt_data:
            raise ValueError(f"TXT data missing required field: {field}")
    
    standardized = {
        'category': normalize_category(txt_data['category']),
        'confidence': normalize_confidence(txt_data['confidence']),
        'user_input': str(txt_data.get('user_input', 'From TXT file'))
    }
    
    # Validate
    is_valid, error = validate_classification(standardized)
    if not is_valid:
        raise ValueError(f"Adapted TXT data is invalid: {error}")
    
    return standardized


def adapt_any_source(data: Dict[str, Any], source_type: str = 'auto') -> Dict[str, Any]:
    """
    Universal adapter - automatically detects source type and adapts
    
    Args:
        data: Raw classification data from any source
        source_type: 'auto', 'llm', 'mock', or 'txt'
    
    Returns:
        dict: Standardized classification data
    """
    
    if source_type == 'auto':
        # Try to detect source type from data structure
        if 'scenario_name' in data:
            source_type = 'txt'
        elif any(key in data for key in ['intent', 'certainty']):
            source_type = 'llm'
        else:
            source_type = 'mock'
    
    # Route to appropriate adapter
    if source_type == 'llm':
        return adapt_llm_output(data)
    elif source_type == 'txt':
        return adapt_txt_output(data)
    else:  # mock or unknown
        return adapt_mock_output(data)


# =====================================================
# Helper Functions for Subgroup A
# =====================================================

def create_llm_response_template() -> Dict[str, Any]:
    """
    Template for what Subgroup A's LLM should return
    
    Returns:
        dict: Empty template with correct structure
    """
    
    return {
        'category': '',       # Required: One of the 57 categories
        'confidence': 0.0,    # Required: 0.0-1.0 or 0-100
        'user_input': '',     # Required: Summary of user's input
        
        # Optional fields (adapter will work without these)
        'symptoms': '',       # Optional: Extracted symptoms
        'urgency': '',        # Optional: crisis/urgent/routine
        'language': 'en'      # Optional: Language detected
    }


def validate_llm_integration(llm_function):
    """
    Test helper to validate that Subgroup A's LLM function works correctly
    
    Args:
        llm_function: The LLM classification function to test
    
    Returns:
        bool: True if LLM function is compatible
    """
    
    # Test conversation
    test_conversation = [
        {'role': 'USER', 'message': 'I need help with anxiety'}
    ]
    
    try:
        # Call their function
        result = llm_function(test_conversation)
        
        # Try to adapt it
        adapted = adapt_llm_output(result)
        
        print("✓ LLM integration test PASSED")
        print(f"  Original output: {result}")
        print(f"  Adapted output: {adapted}")
        
        return True
        
    except Exception as e:
        print(f"✗ LLM integration test FAILED: {e}")
        return False


# =====================================================
# Test Code
# =====================================================

if __name__ == "__main__":
    
    print("="*70)
    print("DATA ADAPTER - TEST SUITE")
    print("="*70)
    
    # Test 1: Mock classifier output
    print("\n[Test 1] Mock Classifier Output")
    mock_output = {
        'category': 'mental health',
        'confidence': 0.85,
        'user_input': 'I need help with depression'
    }
    
    try:
        adapted = adapt_mock_output(mock_output)
        print(f"✓ Input: {mock_output}")
        print(f"✓ Output: {adapted}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 2: LLM output (various formats)
    print("\n[Test 2] LLM Output - Format A")
    llm_output_a = {
        'intent': 'depression',
        'certainty': 85,  # Note: percentage instead of decimal
        'symptoms': 'feeling down for weeks'
    }
    
    try:
        adapted = adapt_llm_output(llm_output_a)
        print(f"✓ Input: {llm_output_a}")
        print(f"✓ Output: {adapted}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n[Test 3] LLM Output - Format B")
    llm_output_b = {
        'category': 'Crisis counseling',
        'confidence': 0.95,
        'user_text': 'I need urgent help'
    }
    
    try:
        adapted = adapt_llm_output(llm_output_b)
        print(f"✓ Input: {llm_output_b}")
        print(f"✓ Output: {adapted}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 4: TXT file output
    print("\n[Test 4] TXT File Output")
    txt_output = {
        'category': 'Mental health',
        'confidence': 0.92,
        'user_input': 'I need affordable therapy',
        'scenario_name': 'Test scenario'
    }
    
    try:
        adapted = adapt_txt_output(txt_output)
        print(f"✓ Input: {txt_output}")
        print(f"✓ Output: {adapted}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 5: Auto-detection
    print("\n[Test 5] Auto-Detection")
    auto_test = {
        'intent': 'anxiety',
        'certainty': 78,
        'text': 'panic attacks'
    }
    
    try:
        adapted = adapt_any_source(auto_test, source_type='auto')
        print(f"✓ Input: {auto_test}")
        print(f"✓ Output: {adapted}")
        print(f"✓ Detected as: LLM format")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 6: Validation catches errors
    print("\n[Test 6] Validation - Missing Field")
    bad_data = {
        'category': 'Mental health'
        # Missing confidence and user_input
    }
    
    try:
        adapted = adapt_mock_output(bad_data)
        print(f"✗ Should have failed validation")
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    # Test 7: Confidence normalization
    print("\n[Test 7] Confidence Normalization")
    test_cases = [
        ({'category': 'Test', 'confidence': 85, 'user_input': 'test'}, "85 → 0.85"),
        ({'category': 'Test', 'confidence': 0.85, 'user_input': 'test'}, "0.85 → 0.85"),
        ({'category': 'Test', 'confidence': '75', 'user_input': 'test'}, "'75' → 0.75"),
    ]
    
    for test_data, description in test_cases:
        adapted = adapt_mock_output(test_data)
        print(f"✓ {description}: {adapted['confidence']}")
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)