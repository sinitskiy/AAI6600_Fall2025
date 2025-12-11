#!/usr/bin/env python3
"""
Anti-Hallucination Validation Module - ENHANCED VERSION

This module prevents unreliable or fabricated facility recommendations by:
1. Validating scoring reliability based on semantic similarity
2. Flagging low-confidence results with clear warnings
3. Adding data source traceability for transparency
4. Providing comprehensive warning reports and disclaimers
5. NEW: Detecting negative evidence that disqualifies facilities

Version: 1.1 - Added Negative Evidence Detection
Author: Group 3 Team
Last Updated: October 2025

Usage:
    from anti_hallucination import AntiHallucinationValidator
    
    validator = AntiHallucinationValidator()
    validated = validator.validate_results(facilities)
    print(validator.generate_warning_report(validated))

Requirements:
    - pandas
    - numpy
    
Note: This module works independently and has no absolute path dependencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re


class NegativeEvidenceDetector:
    """
    Detects explicit evidence that a facility is NOT affordable
    or NOT appropriate for users seeking affordable care.
    
    This class prevents high scores for facilities that explicitly
    state they don't accept insurance or are premium/luxury services.
    """
    
    def __init__(self):
        """Initialize negative evidence patterns"""
        
        # Strong indicators that facility is NOT affordable (instant disqualification)
        self.strong_negative_indicators = [
            "cash only", "private pay only", "no insurance accepted",
            "no medicaid", "no medicare", "self-pay only", "self pay only",
            "does not accept insurance", "insurance not accepted",
            "concierge medicine", "concierge practice", "executive health",
            "luxury clinic", "premium service", "vip treatment",
            "membership required", "members only", "subscription based",
            "direct primary care", "retainer based", "annual fee required"
        ]
        
        # Moderate indicators suggesting potential affordability barriers
        self.moderate_negative_indicators = [
            "private practice", "out of network", "limited insurance",
            "waitlist closed", "not accepting new patients",
            "referral required", "invitation only",
            "sliding scale not available", "no sliding scale",
            "full fee", "standard rates apply", "market rate",
            "boutique", "exclusive", "high-end"
        ]
        
        # Phrases that might seem negative but actually indicate affordability
        self.false_negative_patterns = [
            "no one turned away", "no insurance required",
            "no insurance necessary", "no referral needed",
            "no membership fees", "no hidden costs"
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase and remove extra spaces
        text = str(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def detect_negative_evidence(self, facility_data: Dict) -> Dict:
        """
        Detect negative evidence in facility data
        
        Args:
            facility_data: Dictionary with facility information
            
        Returns:
            Dictionary with detection results:
            - has_strong_negative: bool
            - has_moderate_negative: bool
            - negative_score: float (0-1, higher = more negative evidence)
            - found_indicators: list of detected phrases
            - confidence: float (0-1)
        """
        
        # Combine all text fields for analysis
        text_fields = ['name', 'description', 'combined_text', 'context', 
                      'services', 'notes', 'org']
        
        combined_text = ""
        for field in text_fields:
            if field in facility_data:
                combined_text += " " + str(facility_data.get(field, ""))
        
        combined_text = self.clean_text(combined_text)
        
        if not combined_text:
            return {
                'has_strong_negative': False,
                'has_moderate_negative': False,
                'negative_score': 0.0,
                'found_indicators': [],
                'confidence': 0.0
            }
        
        # Check for false negatives first (actually positive indicators)
        for pattern in self.false_negative_patterns:
            if pattern in combined_text:
                # These phrases actually indicate affordability
                return {
                    'has_strong_negative': False,
                    'has_moderate_negative': False,
                    'negative_score': 0.0,
                    'found_indicators': [],
                    'confidence': 0.9
                }
        
        # Check for strong negative indicators
        found_strong = []
        for indicator in self.strong_negative_indicators:
            if indicator in combined_text:
                found_strong.append(indicator)
        
        # Check for moderate negative indicators
        found_moderate = []
        for indicator in self.moderate_negative_indicators:
            if indicator in combined_text:
                found_moderate.append(indicator)
        
        # Calculate negative evidence score
        has_strong = len(found_strong) > 0
        has_moderate = len(found_moderate) > 0
        
        # Strong indicators are weighted more heavily
        negative_score = min(1.0, 
                           (len(found_strong) * 0.5) + 
                           (len(found_moderate) * 0.2))
        
        # Confidence based on amount of text available
        text_length = len(combined_text.split())
        confidence = min(1.0, text_length / 50)  # Full confidence at 50+ words
        
        return {
            'has_strong_negative': has_strong,
            'has_moderate_negative': has_moderate,
            'negative_score': negative_score,
            'found_indicators': found_strong + found_moderate,
            'confidence': confidence
        }
    
    def adjust_score_for_negative_evidence(self, original_score: float, 
                                          negative_result: Dict) -> float:
        """
        Adjust facility score based on negative evidence
        
        Args:
            original_score: Original facility score (0-10)
            negative_result: Result from detect_negative_evidence
            
        Returns:
            Adjusted score (capped if negative evidence found)
        """
        
        if negative_result['has_strong_negative']:
            # Strong negative evidence caps score at 3.0
            return min(original_score, 3.0)
        elif negative_result['has_moderate_negative']:
            # Moderate negative evidence caps score at 5.0
            # and applies a reduction based on negative score
            reduction = negative_result['negative_score'] * 2.0
            return min(original_score, max(5.0, original_score - reduction))
        else:
            # No significant negative evidence
            return original_score


class AntiHallucinationValidator:
    """Enhanced Anti-Hallucination Validator with Negative Evidence Detection"""
    
    # Confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        'high': 0.70,      # High confidence: similarity > 0.70
        'medium': 0.50,    # Medium confidence: 0.50-0.70
        'low': 0.30        # Low confidence: < 0.50
    }
    
    # Score reasonability thresholds
    SCORE_THRESHOLDS = {
        'suspiciously_high': 9.5,  # Suspiciously high score
        'suspiciously_low': 2.0    # Suspiciously low score
    }
    
    def __init__(self):
        """Initialize validator with negative evidence detector"""
        self.validation_warnings = []
        self.negative_detector = NegativeEvidenceDetector()
    
    def validate_facility(self, facility: Dict) -> Dict:
        """
        Validate reliability of a single facility with negative evidence checking
        
        Args:
            facility: facility dictionary
        
        Returns:
            facility dictionary with added validation information
        """
        
        validation = {
            'confidence_level': 'unknown',
            'warnings': [],
            'data_source': facility.get('source', 'unknown'),
            'reliability_score': 0.0,
            'has_negative_evidence': False,
            'adjusted_score': facility.get('score', 0.0)
        }
        
        # NEW: Check for negative evidence first
        negative_result = self.negative_detector.detect_negative_evidence(facility)
        
        if negative_result['has_strong_negative']:
            validation['warnings'].append(
                f"⚠️ STRONG WARNING: Facility appears to be cash-only or luxury service. "
                f"Found: {', '.join(negative_result['found_indicators'][:2])}"
            )
            validation['has_negative_evidence'] = True
            validation['confidence_level'] = 'low'
            
            # Adjust the facility score
            original_score = facility.get('score', 0.0)
            adjusted_score = self.negative_detector.adjust_score_for_negative_evidence(
                original_score, negative_result
            )
            facility['score'] = adjusted_score
            validation['adjusted_score'] = adjusted_score
            
            if original_score > adjusted_score:
                validation['warnings'].append(
                    f"Score reduced from {original_score:.1f} to {adjusted_score:.1f} "
                    f"due to negative evidence"
                )
        
        elif negative_result['has_moderate_negative']:
            validation['warnings'].append(
                f"⚠️ Potential barriers: {', '.join(negative_result['found_indicators'][:2])}"
            )
            validation['has_negative_evidence'] = True
            
            # Apply moderate adjustment
            original_score = facility.get('score', 0.0)
            adjusted_score = self.negative_detector.adjust_score_for_negative_evidence(
                original_score, negative_result
            )
            facility['score'] = adjusted_score
            validation['adjusted_score'] = adjusted_score
        
        # Continue with original validation checks
        # 1. Check score confidence
        confidence = self._check_score_confidence(facility)
        if validation['confidence_level'] != 'low':  # Don't override if already low
            validation['confidence_level'] = confidence
        
        # 2. Check data completeness
        completeness = self._check_data_completeness(facility)
        if completeness < 0.5:
            validation['warnings'].append(
                f"Incomplete data ({completeness*100:.0f}%)"
            )
        
        # 3. Check score anomalies (with adjusted score)
        score_anomalies = self._check_score_anomalies(facility)
        validation['warnings'].extend(score_anomalies)
        
        # 4. Calculate overall reliability score (now includes negative evidence)
        reliability_components = []
        
        # Confidence weight (25%)
        if confidence == 'high':
            reliability_components.append(25.0)
        elif confidence == 'medium':
            reliability_components.append(15.0)
        else:
            reliability_components.append(5.0)
        
        # Completeness weight (25%)
        reliability_components.append(completeness * 25.0)
        
        # No anomalies weight (25%)
        if len(score_anomalies) == 0:
            reliability_components.append(25.0)
        elif len(score_anomalies) == 1:
            reliability_components.append(15.0)
        else:
            reliability_components.append(5.0)
        
        # NEW: No negative evidence weight (25%)
        if not negative_result['has_strong_negative']:
            if not negative_result['has_moderate_negative']:
                reliability_components.append(25.0)
            else:
                reliability_components.append(10.0)
        else:
            reliability_components.append(0.0)
        
        validation['reliability_score'] = sum(reliability_components)
        
        # Add validation results to facility
        facility['validation'] = validation
        
        return facility
    
    def _check_score_confidence(self, facility: Dict) -> str:
        """
        Check confidence level based on score and similarity
        
        Returns:
            'high', 'medium', or 'low'
        """
        
        # Get similarity score if available
        similarity = facility.get('similarity_score', 0)
        
        if similarity > self.CONFIDENCE_THRESHOLDS['high']:
            return 'high'
        elif similarity > self.CONFIDENCE_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _check_data_completeness(self, facility: Dict) -> float:
        """
        Check data completeness
        
        Returns:
            completeness score (0.0 - 1.0)
        """
        
        required_fields = ['name', 'street', 'city', 'state']
        optional_fields = ['zip', 'phone', 'description']
        
        # Check required fields
        complete_required = sum(
            1 for field in required_fields 
            if facility.get(field) and str(facility[field]).strip() 
            and str(facility[field]).lower() != 'nan'
            and 'not available' not in str(facility[field]).lower()
        )
        
        complete_optional = 0
        for field in optional_fields:
            value = facility.get(field, '')
            # Check if valid value (not empty, nan, or "not available")
            if (value and str(value).strip() 
                and str(value).lower() not in ['nan', 'address not available', 
                                                 'phone not available', '']):
                complete_optional += 1
        
        # Required fields weight 80%, optional fields weight 20%
        completeness = (
            (complete_required / len(required_fields)) * 0.8 +
            (complete_optional / len(optional_fields)) * 0.2
        )
        
        return completeness
    
    def _check_score_anomalies(self, facility: Dict) -> List[str]:
        """
        Check score anomalies
        
        Returns:
            list of warning messages
        """
        
        warnings = []
        
        overall_score = facility.get('score', 0)
        
        # Check suspiciously high scores
        if overall_score >= self.SCORE_THRESHOLDS['suspiciously_high']:
            warnings.append(
                f"Abnormally high score ({overall_score:.1f}/10), manual verification recommended"
            )
        
        # Check suspiciously low scores
        if overall_score <= self.SCORE_THRESHOLDS['suspiciously_low']:
            warnings.append(
                f"Abnormally low score ({overall_score:.1f}/10), may not match criteria"
            )
        
        # Check consistency across five dimensions
        dimension_scores = []
        for dim in ['affordability_score', 'crisis_care_score', 
                   'accessibility_score', 'specialization_score', 
                   'community_integration_score']:
            if dim in facility:
                dimension_scores.append(facility[dim])
        
        if len(dimension_scores) >= 3:
            std_dev = np.std(dimension_scores)
            if std_dev > 2.0:
                warnings.append(
                    f"Inconsistent scores across dimensions (std dev: {std_dev:.1f})"
                )
        
        return warnings
    
    def validate_results(self, facilities: List[Dict]) -> List[Dict]:
        """
        Validate a list of facilities
        
        Args:
            facilities: list of facility dictionaries
        
        Returns:
            list of validated facility dictionaries
        """
        
        validated = []
        self.validation_warnings = []  # Reset warnings
        
        for facility in facilities:
            validated_facility = self.validate_facility(facility)
            validated.append(validated_facility)
            
            # Collect serious warnings
            if validated_facility.get('validation', {}).get('warnings'):
                self.validation_warnings.append({
                    'facility': validated_facility.get('name', 'Unknown'),
                    'warnings': validated_facility['validation']['warnings']
                })
        
        return validated
    
    def generate_warning_report(self, validated_facilities: List[Dict]) -> str:
        """
        Generate comprehensive warning report with negative evidence summary
        
        Args:
            validated_facilities: list of validated facilities
        
        Returns:
            formatted warning report string
        """
        
        report = []
        report.append("\n" + "="*70)
        report.append("🔍 VALIDATION REPORT")
        report.append("="*70 + "\n")
        
        # Count facilities by confidence level
        confidence_counts = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
        negative_evidence_count = 0
        
        for facility in validated_facilities:
            validation = facility.get('validation', {})
            level = validation.get('confidence_level', 'unknown')
            confidence_counts[level] += 1
            
            if validation.get('has_negative_evidence'):
                negative_evidence_count += 1
        
        # Summary statistics
        total = len(validated_facilities)
        report.append("📊 Summary:")
        report.append(f"   Total facilities: {total}")
        report.append(f"   High confidence: {confidence_counts['high']} ({confidence_counts['high']/total*100:.1f}%)")
        report.append(f"   Medium confidence: {confidence_counts['medium']} ({confidence_counts['medium']/total*100:.1f}%)")
        report.append(f"   Low confidence: {confidence_counts['low']} ({confidence_counts['low']/total*100:.1f}%)")
        
        # NEW: Negative evidence summary
        if negative_evidence_count > 0:
            report.append(f"\n⚠️  Negative Evidence Detected:")
            report.append(f"   {negative_evidence_count} facilities may not accept insurance or have affordability barriers")
            report.append(f"   Scores have been automatically adjusted for these facilities")
        
        # Reliability distribution
        report.append(f"\n📈 Reliability Distribution:")
        reliability_ranges = {
            'Excellent (80-100)': 0,
            'Good (60-79)': 0,
            'Fair (40-59)': 0,
            'Poor (0-39)': 0
        }
        
        for facility in validated_facilities:
            score = facility.get('validation', {}).get('reliability_score', 0)
            if score >= 80:
                reliability_ranges['Excellent (80-100)'] += 1
            elif score >= 60:
                reliability_ranges['Good (60-79)'] += 1
            elif score >= 40:
                reliability_ranges['Fair (40-59)'] += 1
            else:
                reliability_ranges['Poor (0-39)'] += 1
        
        for range_name, count in reliability_ranges.items():
            report.append(f"   {range_name}: {count} facilities")
        
        # Specific warnings
        if self.validation_warnings:
            report.append(f"\n⚠️  Facilities with Warnings ({len(self.validation_warnings)}):")
            for i, warning in enumerate(self.validation_warnings[:5], 1):
                report.append(f"\n   {i}. {warning['facility']}")
                for w in warning['warnings'][:2]:
                    report.append(f"      - {w}")
            
            if len(self.validation_warnings) > 5:
                report.append(f"\n   ... and {len(self.validation_warnings) - 5} more")
        
        return "\n".join(report)
    
    def add_disclaimer(self, facilities: List[Dict]) -> str:
        """
        Generate appropriate disclaimer based on validation results
        
        Args:
            facilities: list of validated facilities
            
        Returns:
            disclaimer text
        """
        
        # Check overall reliability
        low_confidence_count = sum(
            1 for f in facilities 
            if f.get('validation', {}).get('confidence_level') == 'low'
        )
        
        has_negative_evidence = any(
            f.get('validation', {}).get('has_negative_evidence') 
            for f in facilities
        )
        
        disclaimer = [
            "\n" + "="*70,
            "📋 IMPORTANT DISCLAIMER",
            "="*70,
            ""
        ]
        
        if has_negative_evidence:
            disclaimer.append(
                "⚠️  Some facilities may have been incorrectly included due to keyword matching."
            )
            disclaimer.append(
                "   Facilities marked with negative evidence warnings may not accept insurance."
            )
            disclaimer.append("")
        
        if low_confidence_count > len(facilities) * 0.3:
            disclaimer.append(
                "⚠️  Many results have low confidence scores. Please verify all information directly."
            )
        
        disclaimer.extend([
            "This system provides reference information only. Always:",
            "✓ Call facilities directly to confirm insurance acceptance",
            "✓ Verify current availability and services",
            "✓ Confirm all costs and payment options",
            "✓ Check if referrals are required",
            "",
            "For mental health emergencies, call 988 (Suicide & Crisis Lifeline) or 911."
        ])
        
        return "\n".join(disclaimer)


# For testing the module independently
if __name__ == "__main__":
    print("Anti-Hallucination Module v1.1 - With Negative Evidence Detection")
    
    # Test negative evidence detector
    detector = NegativeEvidenceDetector()
    
    test_facilities = [
        {
            'name': 'Premium Wellness Center',
            'description': 'Luxury mental health services, cash only, no insurance accepted',
            'score': 8.5
        },
        {
            'name': 'Community Mental Health Center',
            'description': 'Sliding scale fees available, Medicaid accepted, no one turned away',
            'score': 7.2
        },
        {
            'name': 'Executive Therapy Associates',
            'description': 'Concierge mental health practice with membership required',
            'score': 9.1
        }
    ]
    
    print("\nTesting Negative Evidence Detection:")
    print("-" * 50)
    
    for facility in test_facilities:
        result = detector.detect_negative_evidence(facility)
        adjusted_score = detector.adjust_score_for_negative_evidence(
            facility['score'], result
        )
        
        print(f"\nFacility: {facility['name']}")
        print(f"  Original Score: {facility['score']}")
        print(f"  Negative Evidence: {result['has_strong_negative'] or result['has_moderate_negative']}")
        if result['found_indicators']:
            print(f"  Found: {', '.join(result['found_indicators'][:2])}")
        print(f"  Adjusted Score: {adjusted_score:.1f}")
    
    # Test full validation
    print("\n" + "="*50)
    print("Testing Full Validation with Negative Evidence:")
    print("="*50)
    
    validator = AntiHallucinationValidator()
    validated = validator.validate_results(test_facilities)
    
    print(validator.generate_warning_report(validated))
    print(validator.add_disclaimer(validated))