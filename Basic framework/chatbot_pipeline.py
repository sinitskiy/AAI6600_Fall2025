#!/usr/bin/env python3
"""
Mental Health Chatbot Pipeline - Main Entry Point

This script orchestrates the complete chatbot flow:
1. User conversation (via chatbot_interface.py)
2. Classification (mock for now, real LLM later)
3. Routing (via group2_router.py)
4. Facility matching (if Group 3 category)

Author: Subgroup B (Michael & Radhika)
Date: November 2025
"""

import os
import sys
from pathlib import Path

# Add paths for imports
current_file = Path(__file__).resolve()
root_dir = current_file.parent
p1_dir = root_dir / "p1"
integrated_dir = root_dir / "integrated"

sys.path.insert(0, str(p1_dir))
sys.path.insert(0, str(integrated_dir))

# Import existing modules
from group2_router import handle_group2_input
import pandas as pd
import numpy as np
import re
import json
import requests

# =====================================================
# Crisis Detection System (Hybrid: Keywords + Embeddings + Gemini)
# =====================================================
# This system uses a 3-stage approach for optimal crisis detection:
#   Stage 1: Fast keyword screening (< 1ms)
#   Stage 2: Semantic similarity with sentence-transformers (~50ms, local)
#   Stage 3: Gemini API confirmation for edge cases (~1-2s)
#
# Installation: pip install sentence-transformers scikit-learn
# If sentence-transformers is not available, system falls back to keywords + Gemini
# =====================================================

# Global variables for crisis detection
CRISIS_MODEL = None
CRISIS_EMBEDDINGS = None

# Crisis reference phrases for semantic matching
CRISIS_REFERENCE_PHRASES = [
    "I want to kill myself",
    "I'm thinking about suicide",
    "I want to end my life",
    "I'm going to hurt myself",
    "I don't want to be alive anymore",
    "Everyone would be better off without me",
    "I can't take it anymore and want to die",
    "I want to end it all",
    "I'm planning to harm myself",
    "Life isn't worth living",
    "I wish I was dead",
    "I'm going to take my own life"
]

def initialize_crisis_detection():
    """
    Initialize the sentence transformer model for crisis detection.
    Lazy loading - only loads when first needed.
    """
    global CRISIS_MODEL, CRISIS_EMBEDDINGS
    
    if CRISIS_MODEL is not None:
        return  # Already initialized
    
    try:
        from sentence_transformers import SentenceTransformer
        print("[Initializing crisis detection system...]")
        CRISIS_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        CRISIS_EMBEDDINGS = CRISIS_MODEL.encode(CRISIS_REFERENCE_PHRASES)
        print("✓ Crisis detection ready\n")
    except ImportError:
        print("⚠️  Warning: sentence-transformers not installed.")
        print("   Install with: pip install sentence-transformers")
        print("   Falling back to keyword + Gemini detection.\n")
        CRISIS_MODEL = False  # Mark as unavailable
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize crisis detection: {e}")
        print("   Falling back to keyword + Gemini detection.\n")
        CRISIS_MODEL = False


def detect_crisis_semantic(user_message, threshold=0.65):
    """
    Stage 2: Use semantic similarity with sentence transformers.
    
    Args:
        user_message: User's message to analyze
        threshold: Similarity threshold (0-1) for crisis detection
    
    Returns:
        dict with:
        {
            'is_crisis': bool,
            'confidence': float (0-1),
            'matched_phrase': str,
            'method': 'semantic'
        }
    """
    global CRISIS_MODEL, CRISIS_EMBEDDINGS
    
    if CRISIS_MODEL is None:
        initialize_crisis_detection()
    
    if CRISIS_MODEL is False:
        # Model unavailable, return uncertain
        return {
            'is_crisis': None,
            'confidence': 0.0,
            'matched_phrase': None,
            'method': 'semantic_unavailable'
        }
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get embedding for user message
        user_embedding = CRISIS_MODEL.encode([user_message])[0]
        
        # Compute cosine similarity with all crisis references
        similarities = cosine_similarity(
            [user_embedding], 
            CRISIS_EMBEDDINGS
        )[0]
        
        max_similarity = float(np.max(similarities))
        matched_idx = int(np.argmax(similarities))
        
        return {
            'is_crisis': max_similarity > threshold,
            'confidence': max_similarity,
            'matched_phrase': CRISIS_REFERENCE_PHRASES[matched_idx],
            'method': 'semantic'
        }
    except Exception as e:
        print(f"Warning: Semantic crisis detection failed: {e}")
        return {
            'is_crisis': None,
            'confidence': 0.0,
            'matched_phrase': None,
            'method': 'semantic_error'
        }


def detect_crisis_gemini(user_message):
    """
    Stage 3: Use Gemini API for accurate crisis classification.
    
    Args:
        user_message: User's message to analyze
    
    Returns:
        dict with:
        {
            'is_crisis': bool,
            'confidence': float (0-1),
            'crisis_type': str,
            'reasoning': str,
            'method': 'gemini'
        }
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        api_key = config.get("GEMINI_API_KEY")
    except Exception as e:
        return {
            'is_crisis': None,
            'confidence': 0.0,
            'crisis_type': 'unknown',
            'reasoning': f'Config error: {e}',
            'method': 'gemini_error'
        }
    
    if not api_key:
        return {
            'is_crisis': None,
            'confidence': 0.0,
            'crisis_type': 'unknown',
            'reasoning': 'API key missing',
            'method': 'gemini_error'
        }
    
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    prompt = f"""Analyze this message for signs of immediate crisis or self-harm intent.

Message: "{user_message}"

Respond ONLY with JSON (no other text):
{{
  "is_crisis": true/false,
  "confidence": 0-100,
  "crisis_type": "suicide | self-harm | severe_distress | none",
  "reasoning": "brief explanation"
}}

Crisis indicators include:
- Suicidal ideation (wanting to die, ending life, suicide)
- Self-harm intent (hurting oneself, cutting, harming)
- Severe hopelessness with despair
- Giving up on life
- Phrases like "ending it", "can't go on", "want to die"

NOT crisis:
- General sadness or anxiety without self-harm intent
- Seeking help for depression/anxiety
- Feeling stressed or overwhelmed
"""
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }
    params = {"key": api_key}
    
    try:
        response = requests.post(endpoint, headers=headers, params=params, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Extract JSON from response
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            gemini_result = json.loads(match.group(0))
            return {
                'is_crisis': gemini_result.get('is_crisis', False),
                'confidence': gemini_result.get('confidence', 0) / 100.0,
                'crisis_type': gemini_result.get('crisis_type', 'unknown'),
                'reasoning': gemini_result.get('reasoning', ''),
                'method': 'gemini'
            }
        else:
            return {
                'is_crisis': None,
                'confidence': 0.0,
                'crisis_type': 'unknown',
                'reasoning': 'Failed to parse Gemini response',
                'method': 'gemini_parse_error'
            }
    except Exception as e:
        return {
            'is_crisis': None,
            'confidence': 0.0,
            'crisis_type': 'unknown',
            'reasoning': f'Gemini API error: {e}',
            'method': 'gemini_error'
        }


def detect_crisis_hybrid(user_message):
    """
    Hybrid 3-stage crisis detection system.
    
    Stage 1: Fast keyword screening (< 1ms)
    Stage 2: Semantic similarity with embeddings (~50ms, local)
    Stage 3: Gemini confirmation for edge cases (~1-2s, API)
    
    Args:
        user_message: User's message to analyze
    
    Returns:
        dict with:
        {
            'is_crisis': bool,
            'confidence': float (0-1),
            'method': str,
            'details': dict
        }
    """
    message_lower = user_message.lower()
    
    # Stage 1: Fast keyword screening
    urgent_keywords = [
        'kill', 'suicide', 'suicidal', 'die', 'dying', 'dead',
        'hurt myself', 'hurting myself', 'harm myself', 'end my life', 'end it',
        'ending it', 'take my life', 'better off dead', 'better off without',
        "can't go on", "cant go on", 'give up', 'no point', 'want to die',
        'cut myself', 'cutting myself', 'self harm', 'self-harm'
    ]
    
    has_urgent_keyword = any(kw in message_lower for kw in urgent_keywords)
    
    if not has_urgent_keyword:
        # No urgent keywords - very likely not a crisis
        return {
            'is_crisis': False,
            'confidence': 0.95,
            'method': 'keyword_screening',
            'details': {'stage': 1, 'matched_keyword': None}
        }
    
    # Stage 2: Semantic similarity (if model available)
    semantic_result = detect_crisis_semantic(user_message, threshold=0.65)
    
    if semantic_result['is_crisis'] is not None:
        # Semantic model worked
        if semantic_result['confidence'] >= 0.65:
            # Trust semantic model at 0.65+ confidence (no need for Gemini)
            return {
                'is_crisis': semantic_result['is_crisis'],
                'confidence': semantic_result['confidence'],
                'method': 'semantic_trusted',
                'details': {
                    'stage': 2,
                    'matched_phrase': semantic_result['matched_phrase']
                }
            }
    
    # Stage 3: Gemini confirmation (only if semantic failed/unavailable)
    print("   [Double-checking with AI for safety...]")
    gemini_result = detect_crisis_gemini(user_message)
    
    if gemini_result['is_crisis'] is not None:
        return {
            'is_crisis': gemini_result['is_crisis'],
            'confidence': gemini_result['confidence'],
            'method': 'gemini_confirmation',
            'details': {
                'stage': 3,
                'crisis_type': gemini_result['crisis_type'],
                'reasoning': gemini_result['reasoning']
            }
        }
    
    # All stages failed - default to safe side (treat as crisis if keywords present)
    return {
        'is_crisis': True,
        'confidence': 0.70,
        'method': 'fallback_safe_default',
        'details': {
            'stage': 'fallback',
            'reason': 'Detected urgent keywords but could not verify with AI'
        }
    }


def display_emergency_resources():
    """
    Display comprehensive emergency mental health resources and hotlines.
    Always available 24/7 for users in crisis or needing immediate support.
    
    Returns:
        str: Formatted emergency resources with contact information
    """
    resources = """
╔════════════════════════════════════════════════════════════════════╗
║                  🆘 EMERGENCY MENTAL HEALTH RESOURCES              ║
╚════════════════════════════════════════════════════════════════════╝

If you are in immediate danger or having thoughts of self-harm:

📞 **National Suicide Prevention Lifeline**
   Call or Text: 988
   Available: 24/7 - Free and Confidential
   Website: 988lifeline.org

💬 **Crisis Text Line**
   Text: HOME to 741741
   Available: 24/7 - Free Crisis Counseling
   Website: crisistextline.org

🚨 **Emergency Services**
   Call: 911
   For immediate emergency assistance

═══════════════════════════════════════════════════════════════════════

🏥 **SAMHSA National Helpline** (Substance Abuse & Mental Health)
   Call: 1-800-662-HELP (4357)
   Available: 24/7 - Free, Confidential
   Treatment referral and information service
   Website: samhsa.gov/find-help/national-helpline

🌐 **National Alliance on Mental Illness (NAMI)**
   Call: 1-800-950-NAMI (6264)
   Text: "NAMI" to 741741
   Available: Monday-Friday, 10am-10pm ET
   Website: nami.org/help

💙 **The Adam Project** (Free Mental Health Provider Directory)
   Website: www.TheAdamProject.org
   1,300+ free mental health providers across America
   Search by location, specialty, and insurance

🌍 **The Trevor Project** (LGBTQ+ Youth Crisis Support)
   Call: 1-866-488-7386
   Text: START to 678-678
   Available: 24/7
   Website: thetrevorproject.org

📱 **Veterans Crisis Line**
   Call: 988 (Press 1)
   Text: 838255
   Available: 24/7 - Confidential support for veterans
   Website: veteranscrisisline.net

🧠 **National Institute of Mental Health (NIMH)**
   Website: nimh.nih.gov
   Information, research, and resources on mental health conditions

🆘 **Disaster Distress Helpline**
   Call: 1-800-985-5990
   Text: "TalkWithUs" to 66746
   Available: 24/7 - Crisis counseling for disaster-related distress

👤 **National Domestic Violence Hotline**
   Call: 1-800-799-7233
   Text: START to 88788
   Available: 24/7
   Website: thehotline.org

🌈 **Trans Lifeline**
   Call: 877-565-8860 (US) or 877-330-6366 (Canada)
   Available: 10am-4am ET
   Peer support by and for transgender people

📚 **MentalHealth.gov** (Government Resource Portal)
   Website: mentalhealth.gov
   Comprehensive directory of mental health services and information

═══════════════════════════════════════════════════════════════════════
Remember: You are not alone. Help is available right now.
═══════════════════════════════════════════════════════════════════════
"""
    return resources


def assess_crisis_severity(user_message, crisis_result):
    """
    Assess the severity level of a crisis situation.
    
    Uses the crisis detection result to determine urgency and provide
    appropriate guidance for the level of support needed.
    
    Args:
        user_message: The user's message that triggered crisis detection
        crisis_result: Result dict from detect_crisis_hybrid()
    
    Returns:
        dict with:
        {
            'severity': str ('immediate' | 'high' | 'moderate' | 'low'),
            'urgency_score': int (1-10),
            'recommended_action': str,
            'details': dict
        }
    """
    message_lower = user_message.lower()
    confidence = crisis_result.get('confidence', 0)
    
    # Immediate danger indicators
    immediate_keywords = [
        'right now', 'tonight', 'today', 'plan to', 'planning', 'going to',
        'about to', 'ready to', 'have a gun', 'have pills', 'overdose',
        'jump', 'hanging', 'cut my wrists'
    ]
    
    # High urgency indicators
    high_urgency_keywords = [
        'want to die', 'wish i was dead', 'better off dead', 'ending it',
        'kill myself', 'suicide', 'take my life', 'end my life'
    ]
    
    # Moderate urgency indicators
    moderate_keywords = [
        'thoughts of', 'thinking about', 'sometimes think', 'crossed my mind',
        "can't go on", 'no point', 'give up', 'hopeless'
    ]
    
    has_immediate = any(kw in message_lower for kw in immediate_keywords)
    has_high_urgency = any(kw in message_lower for kw in high_urgency_keywords)
    has_moderate = any(kw in message_lower for kw in moderate_keywords)
    
    # Determine severity
    if has_immediate:
        severity = 'immediate'
        urgency_score = 10
        recommended_action = 'Call 911 immediately or go to nearest emergency room'
    elif has_high_urgency and confidence > 0.8:
        severity = 'high'
        urgency_score = 8
        recommended_action = 'Call 988 (Suicide Prevention Lifeline) now - available 24/7'
    elif has_high_urgency or (has_moderate and confidence > 0.7):
        severity = 'high'
        urgency_score = 7
        recommended_action = 'Call 988 or text HOME to 741741 for immediate support'
    elif has_moderate:
        severity = 'moderate'
        urgency_score = 5
        recommended_action = 'Contact crisis support (988 or Crisis Text Line) soon'
    else:
        severity = 'low'
        urgency_score = 3
        recommended_action = 'Consider reaching out to mental health professional'
    
    return {
        'severity': severity,
        'urgency_score': urgency_score,
        'recommended_action': recommended_action,
        'details': {
            'has_immediate_danger': has_immediate,
            'has_high_urgency': has_high_urgency,
            'has_moderate_concern': has_moderate,
            'confidence': confidence,
            'method': crisis_result.get('method', 'unknown')
        }
    }


def display_free_resource_info():
    """
    Display information about free and low-cost mental health resources.
    
    Helpful for users who may not have insurance or are looking for affordable options.
    Highlights TheAdamProject.org as a comprehensive free provider directory.
    
    Returns:
        str: Formatted information about free mental health resources
    """
    info = """
┌────────────────────────────────────────────────────────────────────┐
│                  💙 FREE & AFFORDABLE MENTAL HEALTH CARE           │
└────────────────────────────────────────────────────────────────────┘

If cost is a concern, there are many free and low-cost options available:

🌟 **The Adam Project** - Comprehensive Free Provider Directory
   Website: www.TheAdamProject.org
   • 1,300+ free mental health providers across all 50 states
   • Search by location, specialty, and type of care needed
   • Includes therapists, counselors, psychiatrists, and support groups
   • No insurance required - completely free services

📞 **SAMHSA Treatment Locator**
   Website: findtreatment.gov
   Call: 1-800-662-4357
   • Find local treatment facilities and support groups
   • Many offer sliding scale fees based on income

🏥 **Community Health Centers**
   Website: findahealthcenter.hrsa.gov
   • Federally qualified health centers provide mental health services
   • Sliding scale fees - no one turned away for inability to pay

🎓 **University Training Clinics**
   • Many universities offer low-cost therapy through training programs
   • Supervised by licensed professionals
   • Search: "[your city] university counseling training clinic"

🌐 **Online & Telehealth Resources**
   • BetterHelp Financial Aid: betterhelp.com/get-started
   • Open Path Collective: openpathcollective.org ($30-$80 per session)
   • 7 Cups: 7cups.com (free emotional support)

💬 **Support Groups** (Often Free)
   • NAMI Support Groups: nami.org/support
   • SMART Recovery: smartrecovery.org
   • Mental Health America: mhanational.org/find-support-groups

────────────────────────────────────────────────────────────────────────
Remember: Financial concerns should never prevent you from getting help.
────────────────────────────────────────────────────────────────────────
"""
    return info


def display_online_resources():
    """
    Display comprehensive online mental health resources with direct website links.
    
    Organized by category for easy navigation. Includes telehealth platforms,
    self-help apps, support communities, educational resources, and crisis chat services.
    
    Returns:
        str: Formatted online resources with URLs
    """
    resources = """
┌────────────────────────────────────────────────────────────────────┐
│              🌐 ONLINE MENTAL HEALTH RESOURCES                     │
└────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💻 TELEHEALTH & ONLINE THERAPY PLATFORMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 **BetterHelp** - Online Therapy Platform
   Website: www.betterhelp.com
   • Licensed therapists via video, phone, or messaging
   • Financial aid available for those who qualify
   • Get matched with a therapist in 24-48 hours

🔹 **Talkspace** - Therapy & Psychiatry Online
   Website: www.talkspace.com
   • Therapy and medication management
   • Insurance accepted for many plans
   • Text, video, or audio messaging

🔹 **MDLive** - Telehealth Services
   Website: www.mdlive.com/behavioral-health
   • Psychiatry and therapy services
   • Often covered by insurance
   • Same-day appointments available

🔹 **Cerebral** - Online Mental Health Care
   Website: www.cerebral.com
   • Therapy and medication management
   • Prescriptions delivered to your door
   • Accepts insurance

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📱 MENTAL HEALTH APPS & SELF-HELP TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 **Headspace** - Meditation & Mindfulness
   Website: www.headspace.com
   • Guided meditation and mindfulness exercises
   • Sleep sounds and focus music
   • Stress and anxiety management tools

🔹 **Calm** - Sleep & Meditation App
   Website: www.calm.com
   • Sleep stories and relaxation techniques
   • Breathing exercises and guided meditations
   • Anxiety and stress relief programs

🔹 **Sanvello** - Mental Health Support App
   Website: www.sanvello.com
   • Mood tracking and cognitive behavioral therapy (CBT)
   • Peer support community
   • Premium features with insurance coverage

🔹 **MoodKit** - CBT-Based Mood Improvement
   Website: www.thriveport.com/products/moodkit
   • Evidence-based CBT techniques
   • Mood tracking and thought checker
   • Activities to improve mental health

🔹 **Woebot** - AI Mental Health Ally
   Website: www.woebothealth.com
   • Free AI-powered mental health support
   • Evidence-based CBT conversations
   • Available 24/7

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👥 ONLINE SUPPORT COMMUNITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 **7 Cups** - Free Emotional Support
   Website: www.7cups.com
   • Free, anonymous online chat support
   • Trained volunteer listeners available 24/7
   • Professional therapy available for a fee

🔹 **NAMI Connection** - Peer Support Groups
   Website: www.nami.org/Support-Education/Support-Groups/NAMI-Connection
   • Free peer-led support groups (many virtual)
   • Led by people with lived mental health experience
   • Weekly meetings, no registration required

🔹 **SMART Recovery** - Addiction Support
   Website: www.smartrecovery.org
   • Free online meetings for addiction recovery
   • Science-based, self-empowering approach
   • Multiple meetings daily

🔹 **Depression and Bipolar Support Alliance (DBSA)**
   Website: www.dbsalliance.org/support/chapters-and-support-groups
   • Peer-led support groups (in-person and virtual)
   • Free and confidential
   • Recovery-focused community

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 EDUCATIONAL RESOURCES & INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 **MentalHealth.gov** - Government Resource Portal
   Website: www.mentalhealth.gov
   • Comprehensive mental health information
   • Treatment locator and helplines
   • Resources for families and friends

🔹 **National Institute of Mental Health (NIMH)**
   Website: www.nimh.nih.gov
   • Research-based mental health information
   • Educational materials on all mental health conditions
   • Clinical trials information

🔹 **National Alliance on Mental Illness (NAMI)**
   Website: www.nami.org
   • Mental health education and advocacy
   • Find Your Local NAMI for support
   • Free educational programs and resources

🔹 **Mental Health America (MHA)**
   Website: www.mhanational.org
   • Free mental health screening tools
   • Educational resources and advocacy
   • Community-based resources

🔹 **Psych Central** - Mental Health Information
   Website: www.psychcentral.com
   • Articles and resources on mental health conditions
   • Quizzes and self-assessments
   • Expert-reviewed content

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💬 CRISIS CHAT & TEXT SUPPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔹 **988 Lifeline Chat** - Suicide Prevention Chat
   Website: www.988lifeline.org/chat
   • Free, confidential crisis chat
   • Available 24/7
   • Connect with trained crisis counselor

🔹 **Crisis Text Line** - Text HOME to 741741
   Website: www.crisistextline.org
   • Free 24/7 crisis support via text
   • Trained crisis counselors
   • All issues welcome, not just suicide

🔹 **IMAlive** - Online Crisis Chat
   Website: www.imalive.org
   • Free online crisis chat service
   • Staffed by trained volunteers
   • Available when you need support

🔹 **Veterans Crisis Line Chat**
   Website: www.veteranscrisisline.net/get-help-now/chat
   • 24/7 confidential chat for veterans
   • Also available via phone (988, press 1) or text (838255)
   • Specialized support for veterans and their families

────────────────────────────────────────────────────────────────────────
💡 TIP: Many of these resources can be accessed from your phone, tablet,
   or computer. Save your favorites for easy access when you need support.
────────────────────────────────────────────────────────────────────────
"""
    return resources


def display_followup_support(user_name, severity=None, has_insurance=False):
    """
    Display follow-up support recommendations based on user's situation.
    
    Provides severity-based timing recommendations and actionable next steps
    to encourage continued engagement with mental health care.
    
    Args:
        user_name: User's name for personalization
        severity: Crisis severity level ('immediate', 'high', 'moderate', 'low', or None)
        has_insurance: Whether user has health insurance
    
    Returns:
        str: Formatted follow-up support message
    """
    # Determine follow-up timing based on severity
    if severity == 'immediate':
        timing = "TODAY - within the next few hours"
        urgency_icon = "🚨"
        priority = "URGENT"
    elif severity == 'high':
        timing = "within 24-48 hours"
        urgency_icon = "⚠️"
        priority = "HIGH PRIORITY"
    elif severity == 'moderate':
        timing = "within the next 3-5 days"
        urgency_icon = "📌"
        priority = "IMPORTANT"
    else:
        timing = "within the next week"
        urgency_icon = "💙"
        priority = "RECOMMENDED"
    
    message = f"""
╔════════════════════════════════════════════════════════════════════╗
║              {urgency_icon} FOLLOW-UP SUPPORT & NEXT STEPS               ║
╚════════════════════════════════════════════════════════════════════╝

{user_name}, you've taken an important first step by reaching out today.
Here's what we recommend for your continued care:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📅 RECOMMENDED TIMELINE: {timing}
   Priority Level: {priority}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ **Immediate Actions:**
   1. Save the facility contacts provided above
   2. Call at least 2-3 facilities to check availability
   3. Ask about their first appointment availability
   4. Confirm they accept your insurance (if applicable)

✅ **Preparing for Your First Appointment:**
   • Write down symptoms, concerns, and questions
   • List any medications you're currently taking
   • Bring your insurance card and photo ID
   • Arrive 10-15 minutes early for paperwork
   • Be honest and open - therapists are there to help, not judge

✅ **While You Wait for Your Appointment:**
   • Use the online resources and apps provided above
   • Reach out to crisis support (988) if you need immediate help
   • Practice self-care: sleep, nutrition, gentle exercise
   • Stay connected with supportive friends or family
   • Consider joining an online support group

✅ **If You Can't Get an Appointment Right Away:**
   • Ask to be placed on a cancellation waiting list
   • Try multiple providers from the list above
   • Contact your insurance for additional in-network providers
   • Consider telehealth options (BetterHelp, Talkspace, Cerebral)
"""

    # Add insurance-specific guidance
    if has_insurance:
        message += """
   • Call your insurance's behavioral health line for assistance
   • Ask about Employee Assistance Programs (EAP) if you're employed
"""
    else:
        message += """
   • Check TheAdamProject.org for additional free providers
   • Contact community health centers (sliding scale fees)
   • Look into university training clinics (low-cost services)
"""

    message += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📞 REMEMBER: You can always call these crisis lines 24/7:
   • 988 - Suicide & Crisis Lifeline
   • Text HOME to 741741 - Crisis Text Line
   • 911 - Emergency services
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💙 {user_name}, getting help is a sign of strength, not weakness.
   You deserve support, and it's out there. Don't give up - keep reaching out.

📧 Consider saving this information or taking a screenshot for your records.
────────────────────────────────────────────────────────────────────────
"""
    return message


# =====================================================
# State Mapping and Location Parsing
# =====================================================

import json

# Mapping full state names (lowercase) to 2-letter codes
STATE_MAPPING = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
    'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI', 'idaho': 'ID',
    'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
    'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
    'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI', 'south carolina': 'SC',
    'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX', 'utah': 'UT',
    'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA', 'west virginia': 'WV',
    'wisconsin': 'WI', 'wyoming': 'WY'
}


def parse_location_input(location_string):
    """
    Parse a location string that may contain city and/or state in various formats.
    
    Handles formats like:
    - "Charlotte North Carolina"
    - "Charlotte, NC"
    - "charlotte nc"
    - "New York, NY"
    - "San Francisco California"
    
    Args:
        location_string: User input string containing city and/or state
    
    Returns:
        tuple: (city, state_code) where state_code is 2-letter abbreviation or None
    """
    if not location_string:
        return None, None
    
    # Normalize input
    input_lower = location_string.lower().strip()
    original_input = location_string.strip()
    
    # First, check for explicit state abbreviations (most reliable)
    # Split by common delimiters and check last token(s)
    tokens = input_lower.replace(',', ' ').split()
    found_state = None
    remaining_text = input_lower
    
    if tokens:
        # Check if last token is a 2-letter state abbreviation
        last_token = tokens[-1].upper()
        if len(last_token) == 2 and last_token.isalpha():
            # Check if it's a valid state code
            if last_token in STATE_MAPPING.values():
                found_state = last_token
                # Remove the abbreviation from remaining text
                remaining_text = ' '.join(tokens[:-1]).strip()
    
    # If no abbreviation found, try to find full state names
    # Check multi-word states first (longest match first to avoid partial matches)
    if not found_state:
        sorted_states = sorted(STATE_MAPPING.items(), key=lambda x: len(x[0]), reverse=True)
        
        for full_name, abbrev in sorted_states:
            # For states that could be city names (like "New York"), 
            # only match if it appears at the end of the string
            if full_name in ['new york', 'washington']:
                # Check if state name appears at the end (after city)
                if input_lower.endswith(full_name) or input_lower.endswith(f', {full_name}'):
                    found_state = abbrev
                    # Remove the state from end
                    remaining_text = input_lower.replace(full_name, '').strip()
                    break
            else:
                # For other states, match anywhere
                if full_name in input_lower:
                    found_state = abbrev
                    # Remove the state from input to extract city
                    remaining_text = input_lower.replace(full_name, '').strip()
                    break
    
    # Clean up city name
    city = remaining_text.replace(',', '').strip()
    
    # Capitalize city name properly
    if city:
        # Handle special cases like DC, NYC
        if city.isupper() and len(city) <= 3:
            city = city.upper()
        else:
            # Title case each word
            city = ' '.join(word.capitalize() for word in city.split())
    else:
        city = None
    
    return city, found_state


def validate_us_location(city, state, raw_input=None):
    """
    Validate that the provided location is within the United States.
    
    Checks if the state is a valid US state code and detects common
    non-US location indicators.
    
    Args:
        city: City name (can be None)
        state: State code (2-letter abbreviation)
        raw_input: Original user input (optional, for better detection)
    
    Returns:
        tuple: (is_valid: bool, error_message: str, country_detected: str or None)
    """
    # List of valid US state codes (including DC)
    VALID_US_STATES = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    ]
    
    # Common non-US indicators (countries and provinces/regions)
    NON_US_INDICATORS = {
        'canada': ['ontario', 'quebec', 'british columbia', 'alberta', 'manitoba', 
                   'saskatchewan', 'nova scotia', 'new brunswick', 'montreal', 
                   'toronto', 'vancouver', 'calgary', 'ottawa', 'edmonton', 'canada'],
        'uk': ['london', 'england', 'scotland', 'wales', 'northern ireland', 
               'manchester', 'birmingham', 'liverpool', 'edinburgh', 'glasgow',
               'united kingdom', 'uk'],
        'mexico': ['mexico city', 'guadalajara', 'monterrey', 'cancun', 'tijuana', 'mexico'],
        'australia': ['sydney', 'melbourne', 'brisbane', 'perth', 'adelaide', 'australia'],
        'other': ['paris', 'berlin', 'tokyo', 'beijing', 'dubai', 'singapore']
    }
    
    # Check raw input first (most reliable for detecting country names)
    if raw_input:
        raw_lower = raw_input.lower()
        for country, indicators in NON_US_INDICATORS.items():
            if any(indicator in raw_lower for indicator in indicators):
                country_name = "Canada" if country == "canada" else "the UK" if country == "uk" else country.upper()
                return False, f"I can only search for facilities in the United States.", country_name
    
    # Check if state is invalid
    if state and state.upper() not in VALID_US_STATES:
        # Check if it matches a known non-US state/province code
        non_us_codes = ['ON', 'QC', 'BC', 'AB', 'MB', 'SK', 'NS', 'NB',  # Canada
                        'NSW', 'VIC', 'QLD', 'WA', 'SA']  # Australia
        if state.upper() in non_us_codes:
            return False, "I can only search for facilities in the United States.", "Canada or Australia"
        else:
            return False, "I didn't recognize that state code. Please use a US state (e.g., NC, CA, TX).", None
    
    # Check city for non-US indicators
    if city:
        city_lower = city.lower()
        for country, indicators in NON_US_INDICATORS.items():
            if any(indicator in city_lower for indicator in indicators):
                country_name = "Canada" if country == "canada" else "the UK" if country == "uk" else country.upper()
                return False, f"I can only search for facilities in the United States.", country_name
    
    # Valid US location
    return True, "", None


def fast_search_scored_csv(scored_csv_path, city=None, state=None, zipcode=None, top_n=5):
    """
    Lightweight search over a pre-scored CSV file.

    Purpose:
    - Provide a fast, read-only search path that does NOT instantiate
      FacilityScorer or load any sentence-transformers models.
    - Intended for interactive/demo usage where a scored CSV already
      exists on disk and we want quick filtering/sorting.

    Behavior:
    - Reads `scored_csv_path` via pandas, filters by `state`, `city`, and `zipcode`
      (when provided),
      sorts by `overall_care_needs_score` (if present) and returns up to `top_n`
      records as a list of dictionaries.
    - Falls back to simple alphabetical/available-score sorting when score
      columns are missing.

    NOTE: This is intentionally lightweight and may not capture all of the
    sophisticated filtering/scoring logic present in `FacilityScorer`. Use
    the full scorer for production-quality matching.
    """

    # Read only the necessary columns to reduce memory usage
    usecols = [
        'name', 'street', 'city', 'state', 'zipcode', 'zip', 'phone',
        'overall_care_needs_score', 'affordability_score', 'crisis_care_score'
    ]
    
    try:
        # Read CSV with optimized settings
        df = pd.read_csv(
            scored_csv_path,
            dtype={
                'name': str, 'street': str, 'city': str, 'state': str,
                'zipcode': str, 'zip': str, 'phone': str,
                'overall_care_needs_score': float,
                'affordability_score': float,
                'crisis_care_score': float
            },
            usecols=lambda x: x in usecols,  # Only read needed columns
            na_values=['', 'NA', 'N/A'],  # Handle missing values
            low_memory=True  # Enable memory optimization
        )
    except Exception as e:
        print(f"Warning: Optimized loading failed, falling back to basic load: {e}")
        df = pd.read_csv(scored_csv_path, dtype=str)
        
        # Normalize and coerce numeric score columns if present
        for col in ['overall_care_needs_score', 'affordability_score', 'crisis_care_score']:
            if col in df.columns:
                # coerce to float where possible
                df[col] = pd.to_numeric(df[col], errors='coerce')

    try:
        # Apply filters with error handling
        if state:
            try:
                state_code = state.upper()
                state_mask = df['state'].str.strip().str.upper() == state_code
                df = df[state_mask]
            except Exception as e:
                print(f"Warning: State filtering failed: {e}")

        if city:
            try:
                # Case-insensitive exact match after normalization
                city_normalized = city.lower().strip()
                city_mask = df['city'].str.lower().str.strip() == city_normalized
                df = df[city_mask]
            except Exception as e:
                print(f"Warning: City filtering failed: {e}")

        if zipcode:
            try:
                target_zip = ''.join(ch for ch in str(zipcode) if ch.isdigit())
                if target_zip:
                    df_before_zip = df.copy()
                    zip_mask = pd.Series(False, index=df.index)
                    has_zip_col = False
                    for zcol in ('zipcode', 'zip'):
                        if zcol in df.columns:
                            has_zip_col = True
                            normalized = (
                                df[zcol]
                                .fillna('')
                                .astype(str)
                                    .str.replace(r'\D', '', regex=True)
                            )
                            zip_mask = zip_mask | normalized.str.startswith(target_zip)
                    if has_zip_col:
                        df = df[zip_mask]
                        if df.empty:
                            df = df_before_zip
            except Exception as e:
                print(f"Warning: Zipcode filtering failed: {e}")

        # Sort efficiently using stable sort for consistency
        sort_column = None
        if 'overall_care_needs_score' in df.columns:
            sort_column = 'overall_care_needs_score'
        elif 'affordability_score' in df.columns:
            sort_column = 'affordability_score'
        elif 'name' in df.columns:
            sort_column = 'name'

        if sort_column:
            df = df.nlargest(top_n, sort_column) if sort_column != 'name' else df.nsmallest(top_n, sort_column)
        else:
            df = df.head(top_n)  # Fallback if no sort column found

        # Convert to records efficiently
        records = df.replace({np.nan: None}).to_dict(orient='records')
        
    except Exception as e:
        print(f"Warning: Error during filtering/sorting: {e}")
        records = []
    return records


def format_facility_results(facilities, output_format='simple'):
    """
    Format facility results for end users.

    Args:
        facilities: list of dicts (raw facility records)
        output_format: 'simple' for human-readable text, 'json' for raw JSON

    Returns:
        str: formatted output string
    """
    if not facilities:
        return "No facilities found."

    # Normalize some fields for cleaner output (zipcodes, phone)
    normalized = []
    for f in facilities:
        nf = dict(f) if isinstance(f, dict) else f
        # Normalize zipcode fields to string without trailing .0
        for zkey in ('zip', 'zipcode'):
            if zkey in nf and nf[zkey] is not None:
                try:
                    # handle floats like 62711.0 and numeric strings
                    val = nf[zkey]
                    if isinstance(val, float):
                        nf[zkey] = str(int(val))
                    else:
                        s = str(val)
                        if s.endswith('.0'):
                            nf[zkey] = s[:-2]
                        else:
                            nf[zkey] = s
                except Exception:
                    nf[zkey] = str(nf[zkey])

        # Ensure phone is a string
        if 'phone' in nf and nf['phone'] is not None:
            nf['phone'] = str(nf['phone'])

        normalized.append(nf)

    if output_format == 'json':
        # Return pretty-printed JSON for technical users using normalized data
        return json.dumps(normalized, indent=2, default=str)

    # Simple human-friendly format
    lines = []
    for i, f in enumerate(normalized, 1):
        name = f.get('name') or f.get('facility_name') or f.get('org') or 'Unknown Facility'
        street = f.get('address') or f.get('street') or f.get('Provider First Line Business Practice Location Address', '')
        city = f.get('city') or f.get('city_raw') or ''
        state = f.get('state') or f.get('state_raw') or ''
        zipcode = f.get('zip') or f.get('zipcode') or ''
        phone = f.get('phone') or f.get('telephone') or 'Phone not available'

        # Score fallback
        score = f.get('overall_care_needs_score') or f.get('score')
        try:
            score_str = f"{float(score):.1f}/10" if score is not None else 'N/A'
        except Exception:
            score_str = str(score)

        lines.append(f"{i}. {name} — {street if street else 'Address not available'}")
        loc_line = f"   Location: {city}, {state}"
        if zipcode:
            loc_line += f" {zipcode}"
        lines.append(loc_line)
        lines.append(f"   Phone: {phone}")
        lines.append(f"   Score: {score_str}")
        lines.append("")

    return "\n".join(lines).strip()


# =====================================================
# Mock Classifier (Commented Out, for Reference)
# =====================================================
# def mock_classify_conversation(conversation_history):
#     """
#     LLM-backed conversation handler.
#     Behavior:
#     - If OpenAI python package is available and OPENAI_API_KEY is set, use the Chat API
#       to analyze the provided `conversation_history`, conduct up to a small number
#       of follow-up questions (interactive via input()) to extract missing fields,
#       and return a structured dictionary.
#     - If OpenAI isn't available or API key is missing, fall back to a lightweight
#       heuristic that summarizes user messages (keeps prior mock behavior).
#     Returns:
#         dict: must contain 'category', 'confidence', 'user_input'. May also include
#               optional fields: 'symptoms', 'location' (dict), 'insurance' (dict).
#     """
#     # Gather existing user messages into a single text blob
#     # user_messages = [msg.get('message', '') for msg in conversation_history if msg.get('role') == 'USER']
#     # conversation_text = "\n".join(user_messages).strip()
#     # Fallback heuristic: summarize user messages and return low-confidence classification
#     # combined_input = conversation_text or ""
#     # return {
#     #     'category': 'Mental health',
#     #     'confidence': 0.6,
#     #     'user_input': combined_input[:100],
#     #     'symptoms': combined_input,
#     #     'location': {},
#     #     'insurance': {}
#     # }

# =====================================================
# Harbor Chatbot Helper Functions
# =====================================================

def harbor_greet():
    """
    Harbor introduces itself and asks for the user's name.
    Returns Harbor's greeting message.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        api_key = config.get("GEMINI_API_KEY")
    except Exception as e:
        raise RuntimeError(f"Could not read Gemini API key from config.json: {e}")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in config.json")
    
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    system_prompt = """You are Harbor, a warm and empathetic mental health assistant. Your mission is to help users find the right mental health support.

Start by greeting the user warmly and asking for their name. Keep your greeting brief and friendly (2-3 sentences max).

Example: "Hello! I'm Harbor, and I'm here to help you find the mental health support you need. What's your name?"
"""
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": system_prompt}]
            }
        ]
    }
    params = {"key": api_key}
    
    try:
        response = requests.post(endpoint, headers=headers, params=params, json=payload)
        response.raise_for_status()
        result = response.json()
        greeting = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        return greeting
    except Exception as e:
        # Fallback if API fails
        return "Hello! I'm Harbor, and I'm here to help you find the mental health support you need. What's your name?"


def harbor_ask_concern(user_name, conversation_history):
    """
    After getting the user's name, Harbor asks what's on their mind.
    
    Args:
        user_name: The user's name
        conversation_history: List of conversation messages so far
    
    Returns:
        Harbor's response asking about their concern
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        api_key = config.get("GEMINI_API_KEY")
    except Exception as e:
        raise RuntimeError(f"Could not read Gemini API key from config.json: {e}")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in config.json")
    
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    # Build conversation for Gemini
    prompt = f"""You are Harbor. The user just told you their name is {user_name}. 
    
Respond warmly by:
1. Acknowledging their name
2. Asking what's on their mind or how you can help them today

Keep it brief (2-3 sentences) and empathetic.

Example: "Hi {user_name}, it's nice to meet you. What's on your mind today? How can I help you?"
"""
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]
    }
    params = {"key": api_key}
    
    try:
        response = requests.post(endpoint, headers=headers, params=params, json=payload)
        response.raise_for_status()
        result = response.json()
        message = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        return message
    except Exception as e:
        # Fallback if API fails
        return f"Hi {user_name}, it's nice to meet you. What's on your mind today? How can I help you?"


def harbor_extract_info(conversation_history):
    """
    Extracts structured information from the conversation.
    
    Args:
        conversation_history: List of conversation messages
    
    Returns:
        dict with extracted info and missing fields:
        {
            'user_name': str or None,
            'category': str or None,
            'confidence': int or None,
            'symptoms': str or None,
            'extracted_info': {
                'city': str or None,
                'state': str or None,
                'insurance': str or None,
                'insurance_type': str or None
            },
            'missing_fields': list of field names
        }
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        api_key = config.get("GEMINI_API_KEY")
    except Exception as e:
        raise RuntimeError(f"Could not read Gemini API key from config.json: {e}")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in config.json")
    
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    # Build conversation text
    conversation_text = "\n".join([
        f"{msg.get('role', 'USER')}: {msg.get('message', '')}"
        for msg in conversation_history
    ])
    
    extraction_prompt = f"""Based on the conversation below, extract the following information in JSON format.
If a field is not mentioned, use null.

Conversation:
{conversation_text}

Extract this JSON (respond ONLY with the JSON, no other text):
{{
  "user_name": "name" or null,
  "category": "mental health | substance abuse | general health | crisis" or null,
  "confidence": 0-100 or null,
  "symptoms": "brief description" or null,
  "extracted_info": {{
    "city": "city name" or null,
    "state": "state name or abbreviation" or null,
    "insurance": "yes | no" or null,
    "insurance_type": "provider name" or null
  }},
  "missing_fields": ["list", "of", "missing", "fields"]
}}

IMPORTANT: 
- Only include fields that were explicitly mentioned
- For location: extract both city and state if mentioned (e.g., "I live in Charlotte" → city="Charlotte", state=null)
- For insurance: only mark as "yes" or "no" if explicitly stated
- Missing fields should list: city, state, insurance, insurance_type (if insurance=yes) for any that weren't mentioned
"""
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": extraction_prompt}]
            }
        ]
    }
    params = {"key": api_key}
    
    try:
        response = requests.post(endpoint, headers=headers, params=params, json=payload)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        # Extract JSON from response
        import re
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            extracted = json.loads(match.group(0))
            return extracted
        else:
            # Return default structure if parsing fails
            return {
                'user_name': None,
                'category': None,
                'confidence': None,
                'symptoms': None,
                'extracted_info': {
                    'city': None,
                    'state': None,
                    'insurance': None,
                    'insurance_type': None
                },
                'missing_fields': ['city', 'state', 'insurance']
            }
    except Exception as e:
        print(f"Warning: Info extraction failed: {e}")
        return {
            'user_name': None,
            'category': None,
            'confidence': None,
            'symptoms': None,
            'extracted_info': {
                'city': None,
                'state': None,
                'insurance': None,
                'insurance_type': None
            },
            'missing_fields': ['city', 'state', 'insurance']
        }


def harbor_respond_with_empathy(user_name, user_concern, symptoms, category):
    """
    Provides empathetic acknowledgment and crisis resources when needed.
    
    Uses hybrid crisis detection (keywords + embeddings + Gemini) for accuracy.
    
    Args:
        user_name: User's name
        user_concern: What the user initially shared
        symptoms: Extracted symptoms description
        category: Detected category
    
    Returns:
        dict with:
        {
            'is_crisis': bool,
            'response_given': bool,
            'detection_method': str
        }
    """
    # Use hybrid crisis detection system
    concern_text = f"{user_concern} {symptoms}"
    crisis_result = detect_crisis_hybrid(concern_text)
    
    is_crisis = crisis_result['is_crisis']
    confidence = crisis_result['confidence']
    method = crisis_result['method']
    
    if is_crisis:
        # Assess crisis severity for appropriate response
        severity_assessment = assess_crisis_severity(concern_text, crisis_result)
        severity = severity_assessment['severity']
        urgency_score = severity_assessment['urgency_score']
        
        # Display comprehensive emergency resources immediately
        print("\n" + display_emergency_resources())
        
        # Display severity-specific guidance
        if severity == 'immediate':
            print("\n" + "╔" + "═"*68 + "╗")
            print("║" + " ⚠️  IMMEDIATE DANGER - URGENT ACTION NEEDED ⚠️ ".center(68) + "║")
            print("╚" + "═"*68 + "╝")
            print(f"\n🚨 Urgency Level: {urgency_score}/10 - IMMEDIATE")
            print(f"📞 {severity_assessment['recommended_action']}")
            print("\nIf you cannot call:")
            print("  • Go to your nearest emergency room")
            print("  • Ask someone nearby to help you")
            print("  • Text 911 if available in your area\n")
        elif severity == 'high':
            print("\n" + "┌" + "─"*68 + "┐")
            print("│" + " 🆘 HIGH URGENCY - Please Reach Out Now ".center(68) + "│")
            print("└" + "─"*68 + "┘")
            print(f"\n🚨 Urgency Level: {urgency_score}/10 - HIGH")
            print(f"📞 {severity_assessment['recommended_action']}")
            print("\nYou don't have to face this alone. Help is available right now.\n")
        else:
            print(f"\n🚨 Urgency Level: {urgency_score}/10")
            print(f"📞 {severity_assessment['recommended_action']}\n")
        
        print(f"🚢 Harbor: {user_name}, I'm really glad you reached out to me.")
        print("          What you're feeling is serious, and I want you to know")
        print("          you're not alone. Please use the resources above for")
        print("          immediate support.\n")
        print(f"          I'm also here to help you find ongoing care and support")
        print(f"          near you. Let me ask a few questions so I can connect")
        print(f"          you with the right local resources.\n")
        print("─"*70 + "\n")
        return {
            'is_crisis': True,
            'response_given': True,
            'detection_method': method,
            'confidence': confidence,
            'severity': severity,
            'urgency_score': urgency_score
        }
    
    # Non-crisis but still empathetic acknowledgment
    empathy_messages = {
        'anxiety': f"🚢 Harbor: {user_name}, thank you for sharing that with me. Anxiety can be\n          really overwhelming, and it takes courage to reach out for help.\n          Let me ask a few questions to find the best resources for you.",
        'depression': f"🚢 Harbor: {user_name}, I appreciate you opening up about this. Depression\n          can feel isolating, but you're taking an important step by\n          seeking support. Let me ask a few questions to help you.",
        'substance': f"🚢 Harbor: {user_name}, thank you for trusting me with this. Recognizing you\n          need help with substance use is a brave and important step.\n          Let me ask a few questions to find the best resources for you.",
        'default': f"🚢 Harbor: {user_name}, thank you for sharing what's going on. I'm here to\n          help you find the support you need. Let me ask a few questions."
    }
    
    # Determine which empathy message to use
    concern_lower = concern_text.lower()
    if 'anxi' in concern_lower or 'panic' in concern_lower or 'worry' in concern_lower:
        message = empathy_messages['anxiety']
    elif 'depress' in concern_lower or 'sad' in concern_lower or 'hopeless' in concern_lower:
        message = empathy_messages['depression']
    elif 'substance' in concern_lower or 'alcohol' in concern_lower or 'drug' in concern_lower or 'drinking' in concern_lower:
        message = empathy_messages['substance']
    else:
        message = empathy_messages['default']
    
    print(f"\n{message}\n")
    print("─"*70 + "\n")
    
    return {
        'is_crisis': False,
        'response_given': True,
        'detection_method': method,
        'confidence': confidence
    }


# =====================================================
# Gemini Classifier (Active)
# =====================================================
import json
import requests
def gemini_classify_conversation(conversation_history):
    """
    LLM-backed conversation handler using Gemini API.
    Matches OpenAI function structure: gathers user messages, sends to Gemini, parses and normalizes output.
    Returns dict with category, confidence, user_input, symptoms, location, insurance.

    """
    # Gather user messages into a single text blob
    user_messages = [msg.get('message', '') for msg in conversation_history if msg.get('role') == 'USER']
    conversation_text = "\n".join(user_messages).strip()

    # Load API key from config.json
    try:
        import os
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path) as f:
            config = json.load(f)
        api_key = config.get("GEMINI_API_KEY")
    except Exception as e:
        raise RuntimeError(f"Could not read Gemini API key from config.json: {e}")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in config.json")

    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    prompt = (
        "You are a clinical intake assistant. Your job is to extract the following fields from the user: "
        "category (short label), confidence (0-1 or 0-100), user_input (short summary), symptoms (brief), location {city, state}, insurance {has_insurance: bool, provider: str or empty}. "
        "If ANY required field is missing, ask a concise follow-up question for ONLY that missing field (e.g., if location is missing, ask for city and state; if insurance is missing, ask: 'Do you have health insurance?'). "
        "If the user answers 'yes' to having insurance, you MUST immediately ask for the insurance provider (e.g., 'Who is your insurance provider?' or 'What kind of insurance do you have?') before proceeding. Never skip this step. You MUST always ask the user directly for insurance status if it is missing—never guess, infer, or default any value. Repeat this process until ALL required fields are present. When you have ALL fields, reply with ONLY a valid JSON object containing these fields, with NO extra text, NO Markdown, and NO explanation. Confidence must be numeric.\n\n"
        f"User input: {conversation_text}"
    )
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    params = {"key": api_key}

    try:
        response = requests.post(endpoint, headers=headers, params=params, json=payload)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        print("Gemini API error response:")
        print(e)
        raise RuntimeError(f"Gemini API request failed: {e}")

    # Print raw Gemini response for debugging
    print("Gemini raw API response:")
    print(json.dumps(result, indent=2))

    # Parse Gemini response and normalize output
    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        print(f"Gemini response text: {text}")
        import re
        # Try to extract JSON from anywhere in the response
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            json_str = match.group(0)
            output = json.loads(json_str)
            # Normalize output fields for consistency
            output.setdefault('category', 'Mental health')
            output.setdefault('confidence', 0.6)
            output.setdefault('user_input', conversation_text[:100])
            output.setdefault('symptoms', '')
            output.setdefault('location', {})
            output.setdefault('insurance', {})
            return output
        else:
            # No JSON found, return follow-up question and flag
            return {
                'needs_followup': True,
                'followup_question': text.strip(),
                'category': None,
                'confidence': None,
                'user_input': conversation_text[:100],
                'symptoms': '',
                'location': {},
                'insurance': {}
            }
    except Exception as e:
        raise RuntimeError(f"Gemini API response parsing error: {e}")


# =====================================================
# Pipeline Functions
# =====================================================

def collect_additional_info():
    """
    Collect location and insurance information from user
    
    This function is called AFTER classification, when we know
    it's a Group 3 category and we need facility recommendations.
    
    Returns:
        dict: {
            'location': {'city': str, 'state': str, 'zip': str},
            'insurance': {'has_insurance': bool, 'provider': str}
        }
    """
    
    print("\n" + "="*70)
    print("ADDITIONAL INFORMATION NEEDED")
    print("="*70)
    print("To find the best facilities for you, I need a bit more information.\n")
    
    # ═══════════════════════════════════════════════════════════════
    # YOUR CODE SECTION 1: Collect Location
    # ═══════════════════════════════════════════════════════════════
    # TODO: Implement location collection
    # 
    # Instructions:
    # 1. Ask user for their city
    # 2. Ask user for their state (2-letter code like "NC")
    # 3. Ask user for their ZIP code (optional) with basic validation
    # 4. Store in a dictionary with keys 'city', 'state', and 'zip'
    # 5. Return the location dict
    #
    # Hints:
    # - Use input() to get user responses
    # - .strip() to remove extra whitespace
    # - .upper() for state codes to standardize (NC not nc)
    #
    # Example output format:
    # location = {'city': 'Charlotte', 'state': 'NC', 'zip': '28202'}
    
    location = {}
    
    # YOUR CODE HERE (5-10 lines)
    # Start with: city = input("What city are you in? ").strip()
    
    # Collect location with smart parsing that handles "Charlotte North Carolina" format
    city_raw = input("What city and state are you in? (e.g., Charlotte, NC or Charlotte North Carolina) ").strip()
    
    # Try to parse city and state from the input
    city, state = parse_location_input(city_raw)
    
    # If we didn't get a state, ask for it explicitly
    if not state:
        state_raw = input("What state are you in? (2-letter code or full name, e.g., NC or North Carolina) ").strip()
        
        # Try to parse the state input
        attempts = 0
        while attempts < 2 and not state:
            if not state_raw:
                # empty input -> keep empty and break
                break

            # Check if it's a 2-letter code
            if len(state_raw) == 2 and state_raw.isalpha():
                state = state_raw.upper()
                break

            # Check if it's a full state name
            mapped = STATE_MAPPING.get(state_raw.lower())
            if mapped:
                state = mapped
                break

            # Not recognized: if first attempt, prompt again; otherwise accept uppercase fallback
            attempts += 1
            if attempts < 2:
                state_raw = input("I couldn't recognize that state. Please enter 2-letter code or full state name (or press Enter to skip): ").strip()
            else:
                # fallback: store uppercase of raw input
                state = state_raw.upper() if state_raw else ''
    
    # If we didn't get a city from parsing, ask for it
    if not city:
        city_input = input("What city are you in? ").strip()
        if city_input:
            # Capitalize properly
            if city_input.isupper() and len(city_input) <= 3:
                city = city_input
            else:
                city = ' '.join(word.capitalize() for word in city_input.split())

    zip_attempts = 0
    zip_raw = ''
    zip_code = ''
    while zip_attempts < 3:
        zip_raw = input("What is your ZIP code? (4 or 5 digits, press Enter to skip) ").strip()
        if not zip_raw:
            break

        digits_only = ''.join(ch for ch in zip_raw if ch.isdigit())
        if len(digits_only) in (4, 5):
            zip_code = digits_only
            break

        zip_attempts += 1
        print("Thanks. Please enter a 4- or 5-digit ZIP code, or press Enter to skip.")

    location = {
        'city_raw': city_raw,
        'city': city,
        'state_raw': state_raw,
        'state': state,
        'zip_raw': zip_raw,
        'zip': zip_code
    }

    
    # END YOUR CODE
    # ═══════════════════════════════════════════════════════════════
    
    # ═══════════════════════════════════════════════════════════════
    # YOUR CODE SECTION 2: Collect Insurance
    # ═══════════════════════════════════════════════════════════════
    # TODO: Implement insurance collection
    #
    # Instructions:
    # 1. Ask user if they have insurance (yes/no)
    # 2. If yes, ask for provider name (optional)
    # 3. Store in a dictionary with keys 'has_insurance' and 'provider'
    # 4. Return the insurance dict
    #
    # Hints:
    # - Convert yes/no to boolean (check if 'yes' in answer.lower())
    # - Provider can be optional (empty string if not provided)
    #
    # Example output format:
    # insurance = {'has_insurance': True, 'provider': 'Medicaid'}
    
    # Ask about insurance (simple yes/no and optional provider name)
    has_ins_raw = input("Do you have health insurance? (yes/no) ").strip()
    has_ins = has_ins_raw.lower()
    # Accept more affirmative variants by checking startswith('y')
    has_insurance = has_ins.startswith('y')
    provider = ''
    provider_raw = ''
    if has_insurance:
        provider_raw = input("If yes, what's your insurance provider? (press Enter to skip) ").strip()
        provider = provider_raw

    insurance = {
        'has_insurance': has_insurance,
        'provider': provider,
        'provider_raw': provider_raw
    }
    # ═══════════════════════════════════════════════════════════════
    
    return {
        'location': location,
        'insurance': insurance
    }


def call_facility_matcher(classification, additional_info):
    """
    Call the facility matching system to get recommendations
    
    This connects to your existing facility_scorer.py logic.
    For now, it's a placeholder that will be implemented later.
    
    Args:
        classification: dict from classifier
        additional_info: dict with location and insurance
    
    Returns:
        list: facility recommendations (or None if not implemented yet)
    """
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + "  🏥 FACILITY SEARCH RESULTS  ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print(f"\n📋 Category: {classification['category']}")
    print(f"📊 Confidence: {classification['confidence']:.0%}")

    # Safely extract city/state from additional_info
    city = None
    state = None
    loc = additional_info.get('location', {}) if isinstance(additional_info, dict) else {}
    if isinstance(loc, dict):
        # prefer normalized 'city'/'state', fall back to raw
        city = loc.get('city') or loc.get('city_raw')
        state = loc.get('state') or loc.get('state_raw')
        zip_code = loc.get('zip') or loc.get('zip_raw')
    else:
        zip_code = None

    location_line = f"📍 Location: {city or 'N/A'}, {state or 'N/A'}"
    if zip_code:
        location_line += f" {zip_code}"
    print(location_line)
    print(f"💳 Insurance: {'Yes' if additional_info.get('insurance', {}).get('has_insurance') else 'No'}")
    print()

    scored_csv = root_dir / "Group3_dataset" / "all_facilities_scored.csv"

    if scored_csv.exists():
        print(f"Using pre-scored data: {scored_csv.name}")
        try:
            # Use lightweight fast search to avoid loading heavy ML models
            facilities = fast_search_scored_csv(
                str(scored_csv),
                city=city,
                state=state,
                zipcode=zip_code,
                top_n=5
            )

            print(f"✓ Found {len(facilities)} facilities (top {min(5, len(facilities))})")

            # Determine desired output format(s)
            output_format = additional_info.get('output_format', 'simple')
            
            if output_format == 'both':
                # Show both formats
                print("\n" + "┌" + "─"*68 + "┐")
                print("│" + "  📄 SIMPLE VIEW  ".center(68) + "│")
                print("└" + "─"*68 + "┘")
                formatted_simple = format_facility_results(facilities, output_format='simple')
                print("\n" + formatted_simple)
                
                print("\n" + "┌" + "─"*68 + "┐")
                print("│" + "  📋 JSON VIEW  ".center(68) + "│")
                print("└" + "─"*68 + "┘")
                formatted_json = format_facility_results(facilities, output_format='json')
                print("\n" + formatted_json)
            else:
                # Show single format
                formatted = format_facility_results(facilities, output_format=output_format)
                print("\n" + formatted)

            return facilities
        except Exception as e:
            print(f"Error during facility search: {e}")
            return None

    # No pre-scored CSV found
    print("\n⚠️  No pre-scored dataset found at Group3_dataset/all_facilities_scored.csv.")
    print("Please add a scored CSV to that path or run scoring via integrated/facility_scorer.py")
    # If caller wants to run the full scorer, import it lazily so module import doesn't
    # immediately require heavy ML dependencies (sentence-transformers, etc.).
    try:
        from integrated import facility_scorer
        # Note: the integration path to run the full scorer can be implemented here
        # (e.g., call facility_scorer.score_csv_file or similar). For now, we simply
        # inform the user and return None.
    except Exception:
        # If importing the scorer fails, keep behavior minimal and inform the user.
        pass

    return None


def run_pipeline():
    """
    Main pipeline orchestration with Harbor chatbot
    
    Flow:
    1. Harbor greets user and asks for name
    2. Harbor asks what's on their mind
    3. Extract info from natural conversation
    4. Fall back to hardcoded prompts for missing fields
    5. Confirm and proceed to facility matching
    """
    
    print("\n" + "═"*70)
    print("  🚢 HARBOR - Mental Health Support Assistant".center(70))
    print("  AAI6600 Fall 2025".center(70))
    print("═"*70)
    print("\nWelcome! I'm here to listen and help you find the support you need.")
    print()
    
    # Initialize conversation tracking
    conversation_history = []
    turn_count = 0
    max_turns = 10
    
    # Step 1: Harbor greets and asks for name
    try:
        harbor_greeting = harbor_greet()
        print(f"🚢 Harbor: {harbor_greeting}\n")
        turn_count += 1
        
        conversation_history.append({'role': 'BOT', 'message': harbor_greeting})
        
        # Get user's name with validation
        user_name_input = ""
        while not user_name_input:
            user_name_input = input("You: ").strip()
            if not user_name_input:
                print("🚢 Harbor: I'd love to know your name so I can help you better.\n")
        
        conversation_history.append({'role': 'USER', 'message': user_name_input})
        turn_count += 1
        
        # Extract name from response (simple heuristic)
        # User might say "My name is Sarah" or just "Sarah"
        user_name = user_name_input
        name_match = re.search(r'(?:name is |i\'m |im |call me )([a-zA-Z]+)', user_name_input.lower())
        if name_match:
            user_name = name_match.group(1).capitalize()
        elif ' ' not in user_name_input and len(user_name_input) < 20:
            user_name = user_name_input.capitalize()
        
    except Exception as e:
        print(f"Note: Harbor greeting had an issue ({e}), continuing with fallback...")
        user_name_input = ""
        while not user_name_input:
            user_name_input = input("What's your name? ").strip()
            if not user_name_input:
                print("Please enter your name so I can help you.\n")
        user_name = user_name_input
        conversation_history.append({'role': 'USER', 'message': user_name})
        turn_count += 2
    
    # Step 2: Harbor asks what's on their mind
    print()
    try:
        concern_prompt = harbor_ask_concern(user_name, conversation_history)
        print(f"🚢 Harbor: {concern_prompt}\n")
        turn_count += 1
        
        conversation_history.append({'role': 'BOT', 'message': concern_prompt})
        
        # Get user's concern with validation
        user_concern = ""
        while not user_concern:
            user_concern = input("You: ").strip()
            if not user_concern:
                print("🚢 Harbor: Please share what's on your mind - I'm here to listen and help.\n")
        
        conversation_history.append({'role': 'USER', 'message': user_concern})
        turn_count += 1
        
    except Exception as e:
        print(f"Note: Harbor had an issue ({e}), continuing...")
        user_concern = ""
        while not user_concern:
            user_concern = input(f"Hi {user_name}, what's on your mind today? ").strip()
            if not user_concern:
                print("Please share what's bringing you here today.\n")
        conversation_history.append({'role': 'USER', 'message': user_concern})
        turn_count += 1
    
    print("\n" + "─"*70)
    print("⚙️  Analyzing your needs...")
    print("─"*70 + "\n")
    
    # Step 3: Extract information from conversation
    try:
        extracted = harbor_extract_info(conversation_history)
        
        user_name = extracted.get('user_name') or user_name
        category = extracted.get('category') or 'Mental health'
        confidence = extracted.get('confidence') or 70
        symptoms = extracted.get('symptoms') or user_concern
        
        extracted_location = extracted.get('extracted_info', {})
        city = extracted_location.get('city')
        state = extracted_location.get('state')
        insurance_status = extracted_location.get('insurance')  # "yes" or "no" or None
        insurance_type = extracted_location.get('insurance_type')
        
        missing_fields = extracted.get('missing_fields', [])
        
        # Check if we need clarification (low confidence or missing category)
        needs_clarification = False
        if not category or category.lower() == 'null':
            needs_clarification = True
        elif confidence and confidence < 60:
            needs_clarification = True
        
        # Ask for clarification if needed
        if needs_clarification and turn_count < max_turns:
            print("─"*70)
            print("🚢 Harbor: I want to make sure I understand what you're going through.")
            print("         Can you tell me a bit more? For example:")
            print("         • Are you dealing with anxiety, depression, or mood issues?")
            print("         • Concerns about substance use?")
            print("         • Are you in a crisis situation needing immediate help?")
            print("         • Something else?")
            print("─"*70 + "\n")
            
            clarification = ""
            while not clarification:
                clarification = input("You: ").strip()
                if not clarification:
                    print("🚢 Harbor: Please help me understand what you're experiencing.\n")
            
            conversation_history.append({'role': 'USER', 'message': clarification})
            turn_count += 1
            
            print("\n" + "─"*70)
            print("⚙️  Analyzing with your additional information...")
            print("─"*70 + "\n")
            
            # Re-extract with the additional context
            try:
                extracted = harbor_extract_info(conversation_history)
                category = extracted.get('category') or 'Mental health'
                confidence = extracted.get('confidence') or 70
                symptoms = extracted.get('symptoms') or f"{user_concern}. {clarification}"
                
                # Update extracted location info (user might have mentioned it in clarification)
                extracted_location = extracted.get('extracted_info', {})
                if extracted_location.get('city') and not city:
                    city = extracted_location.get('city')
                if extracted_location.get('state') and not state:
                    state = extracted_location.get('state')
                if extracted_location.get('insurance') and not insurance_status:
                    insurance_status = extracted_location.get('insurance')
                if extracted_location.get('insurance_type') and not insurance_type:
                    insurance_type = extracted_location.get('insurance_type')
                
                missing_fields = extracted.get('missing_fields', [])
            except Exception as e:
                print(f"Note: Had trouble with clarification ({e}), continuing...")
                # Keep original values but update symptoms to include clarification
                symptoms = f"{user_concern}. {clarification}"
        
        # Step 3.5: Empathetic acknowledgment with crisis detection
        empathy_result = harbor_respond_with_empathy(user_name, user_concern, symptoms, category)
        is_crisis = empathy_result.get('is_crisis', False)
        
        # Show what we understood (unless it was already shown in crisis message)
        if not is_crisis:
            print(f"✓ I understand you're looking for help with: {category}")
            if symptoms:
                print(f"✓ You mentioned: {symptoms[:100]}{'...' if len(symptoms) > 100 else ''}")
            print()
            
            # Feature 7: Comprehensive Symptom Assessment (for non-crisis cases)
            # Ask targeted follow-up questions based on category
            print("🚢 Harbor: To find the best support for you, it helps to know a bit more.\n")
            
            symptom_details = {}
            category_lower = category.lower() if category else ""
            
            if 'anxiety' in category_lower or 'panic' in category_lower:
                duration = input("🚢 Harbor: How long have you been experiencing anxiety?\n          (e.g., few weeks, months, years) or press Enter to skip: ").strip()
                if duration:
                    symptom_details['duration'] = duration
                
                triggers = input("🚢 Harbor: Are there specific situations that trigger your anxiety?\n          or press Enter to skip: ").strip()
                if triggers:
                    symptom_details['triggers'] = triggers
                    
            elif 'depression' in category_lower or 'mood' in category_lower:
                duration = input("🚢 Harbor: How long have you been feeling this way?\n          (e.g., few weeks, months, years) or press Enter to skip: ").strip()
                if duration:
                    symptom_details['duration'] = duration
                
                impact = input("🚢 Harbor: Is this affecting your daily activities (work, relationships, sleep)?\n          (yes/no/somewhat) or press Enter to skip: ").strip()
                if impact:
                    symptom_details['daily_impact'] = impact
                    
            elif 'substance' in category_lower or 'addiction' in category_lower:
                substance_type = input("🚢 Harbor: What substance(s) are you concerned about?\n          or press Enter to skip: ").strip()
                if substance_type:
                    symptom_details['substance_type'] = substance_type
                
                seeking_treatment = input("🚢 Harbor: Are you looking for detox, outpatient, or ongoing support?\n          or press Enter to skip: ").strip()
                if seeking_treatment:
                    symptom_details['treatment_preference'] = seeking_treatment
            else:
                # General mental health
                urgency = input("🚢 Harbor: How urgently do you need support?\n          (immediate, within a week, within a month) or press Enter to skip: ").strip()
                if urgency:
                    symptom_details['urgency'] = urgency
            
            # Update symptoms with additional details
            if symptom_details:
                symptoms = f"{symptoms}. Additional details: {', '.join([f'{k}: {v}' for k, v in symptom_details.items()])}"
            
            print()
        
    except Exception as e:
        print(f"Note: Had trouble extracting info ({e}), will ask directly...")
        category = 'Mental health'
        confidence = 70
        symptoms = user_concern
        city = None
        state = None
        insurance_status = None
        insurance_type = None
        missing_fields = ['city', 'state', 'insurance']
        is_crisis = False
    
    # Step 4: Ask for missing information using hardcoded prompts
    if not is_crisis:
        print("\n" + "┌" + "─"*68 + "┐")
        print("│" + " 📋 Step 2: Getting Location & Insurance Details ".center(68) + "│")
        print("└" + "─"*68 + "┘\n")
    
    # Check turn limit
    if turn_count >= max_turns:
        print("(Switching to quick questions to get you help faster)\n")
    
    # Location
    if not city or not state:
        if turn_count < max_turns:
            location_prompt = "🚢 Harbor: To find the best support near you, what city and state are\n          you in? (e.g., Charlotte, NC)\n\nYou: "
            location_input = ""
            while not location_input:
                location_input = input(location_prompt).strip()
                if not location_input:
                    print("🚢 Harbor: I need your location to find resources near you.\n")
            
            conversation_history.append({'role': 'USER', 'message': location_input})
            turn_count += 1
            
            # Parse location
            parsed_city, parsed_state = parse_location_input(location_input)
            if parsed_city:
                city = parsed_city
            if parsed_state:
                state = parsed_state
            
            # Validate US location (pass raw input for better country detection)
            is_valid, error_msg, detected_country = validate_us_location(city, state, location_input)
            
            if not is_valid:
                print(f"\n🚢 Harbor: {error_msg}")
                
                if detected_country:
                    print(f"          It looks like you're in {detected_country}.")
                    print("\n          I specialize in US mental health resources, but here are")
                    print("          some international resources that may help:\n")
                    print("          📞 Crisis Support:")
                    if 'Canada' in detected_country:
                        print("              • Canada Suicide Prevention: 1-833-456-4566")
                        print("              • Crisis Text Line Canada: Text 686868")
                        print("              • Kids Help Phone: 1-800-668-6868")
                    elif 'UK' in detected_country or 'United Kingdom' in detected_country:
                        print("              • Samaritans (UK): 116 123")
                        print("              • Shout (Text): 85258")
                        print("              • Mind UK: mind.org.uk")
                    else:
                        print("              • Befrienders Worldwide: befrienders.org")
                        print("              • International Assoc. for Suicide Prevention: iasp.info")
                    
                    print(f"\n          💙 For local resources, search: '{city} mental health services'")
                    print("          💙 Or visit: findahelpline.com (global directory)\n")
                else:
                    print("          Please provide a US city and state.\n")
                
                # Ask if they want to provide US location instead
                retry = input("🚢 Harbor: Would you like to search for US facilities instead? (yes/no) ").strip().lower()
                
                if retry.startswith('y'):
                    # Re-ask for US location
                    print("\n🚢 Harbor: Great! Let's find US resources for you.\n")
                    location_input = ""
                    while not location_input:
                        location_input = input("          What US city and state? (e.g., Charlotte, NC)\n\nYou: ").strip()
                        if not location_input:
                            print("🚢 Harbor: Please enter a US location.\n")
                    
                    parsed_city, parsed_state = parse_location_input(location_input)
                    if parsed_city:
                        city = parsed_city
                    if parsed_state:
                        state = parsed_state
                    
                    # Validate again
                    is_valid, error_msg, _ = validate_us_location(city, state)
                    if not is_valid:
                        print(f"\n🚢 Harbor: {error_msg}")
                        print("          I'll need a valid US location to continue the search.\n")
                        return {
                            'status': 'invalid_location',
                            'message': 'User provided non-US location after retry'
                        }
                else:
                    print("\n🚢 Harbor: I understand. I hope the international resources help.")
                    print("          Feel free to return if you need US-based resources later.")
                    print("\n          💙 Remember: You deserve support, no matter where you are.\n")
                    return {
                        'status': 'non_us_location',
                        'location': {'city': city, 'state': state},
                        'country': detected_country
                    }
    
    # If still missing, ask individually
    if not city and turn_count < max_turns:
        city = ""
        while not city:
            city = input("🚢 Harbor: What city? ").strip().title()
            if not city:
                print("🚢 Harbor: Please enter your city.\n")
        turn_count += 1
    
    if not state and turn_count < max_turns:
        state_input = ""
        while not state_input:
            state_input = input("🚢 Harbor: What state? (2-letter code or full name) ").strip()
            if not state_input:
                print("🚢 Harbor: Please enter your state.\n")
        # Normalize state
        _, state = parse_location_input(f"City {state_input}")
        if not state:
            state = state_input.upper() if len(state_input) == 2 else state_input
        turn_count += 1
    
    # Insurance
    if not insurance_status and turn_count < max_turns:
        insurance_prompt = "🚢 Harbor: Do you have health insurance? (yes/no)\n\nYou: "
        insurance_input = ""
        while not insurance_input:
            insurance_input = input(insurance_prompt).strip().lower()
            if not insurance_input:
                print("🚢 Harbor: Please answer yes or no.\n")
        
        insurance_status = 'yes' if insurance_input.startswith('y') else 'no'
        conversation_history.append({'role': 'USER', 'message': insurance_input})
        turn_count += 1
        
        # If no insurance, show free resource information
        if insurance_status == 'no':
            print("\n🚢 Harbor: That's okay - there are many free and low-cost options available.")
            print("          Let me share some resources that might help:\n")
            print(display_free_resource_info())
            print("🚢 Harbor: I'll also search for facilities that offer sliding scale fees")
            print("          or accept patients without insurance.\n")
    
    # Insurance type (if they have insurance) - Feature 5: Detailed Insurance Information
    insurance_details = {}
    if insurance_status == 'yes' and not insurance_type and turn_count < max_turns:
        insurance_type_prompt = "🚢 Harbor: What type of insurance? (e.g., Medicaid, Medicare, Blue Cross)\n\nYou: "
        insurance_type = ""
        while not insurance_type:
            insurance_type = input(insurance_type_prompt).strip()
            if not insurance_type:
                print("🚢 Harbor: Please enter your insurance provider name.\n")
        conversation_history.append({'role': 'USER', 'message': insurance_type})
        turn_count += 1
        
        # Additional insurance details for better matching
        print("\n🚢 Harbor: Just a couple more quick questions about your insurance:")
        print("          (These help me find facilities that accept your specific plan)\n")
        
        # Ask about plan type if it's a major provider
        insurance_lower = insurance_type.lower()
        if any(provider in insurance_lower for provider in ['blue cross', 'bcbs', 'aetna', 'cigna', 'unitedhealth', 'united', 'humana']):
            plan_type = input("🚢 Harbor: What type of plan? (e.g., HMO, PPO, EPO) or press Enter to skip: ").strip()
            if plan_type:
                insurance_details['plan_type'] = plan_type
        
        # Ask if they know if behavioral health is covered
        behavioral_coverage = input("🚢 Harbor: Do you know if your plan covers mental health/behavioral health?\n          (yes/no/not sure) or press Enter to skip: ").strip().lower()
        if behavioral_coverage:
            if behavioral_coverage.startswith('y'):
                insurance_details['behavioral_health_covered'] = True
            elif behavioral_coverage.startswith('n'):
                insurance_details['behavioral_health_covered'] = False
            else:
                insurance_details['behavioral_health_covered'] = 'unknown'
        
        print("\n🚢 Harbor: Perfect! This information will help me find the best match.\n")
    
    # Build classification dict
    classification = {
        'category': category,
        'confidence': confidence / 100.0 if confidence > 1 else confidence,
        'user_input': symptoms,
        'symptoms': symptoms,
        'location': {
            'city': city,
            'state': state
        },
        'insurance': {
            'has_insurance': insurance_status == 'yes' if insurance_status else False,
            'provider': insurance_type or '',
            'details': insurance_details if insurance_status == 'yes' else {}
        }
    }
    
    print("\n" + "┌" + "─"*68 + "┐")
    print("│" + " 📍 Information Collected ".center(68) + "│")
    print("└" + "─"*68 + "┘")
    print(f"✓ Location: {city}, {state}")
    
    # Display insurance info with details
    if insurance_status == 'yes' and insurance_type:
        insurance_display = f"Yes - {insurance_type}"
        if insurance_details.get('plan_type'):
            insurance_display += f" ({insurance_details['plan_type']})"
        print(f"✓ Insurance: {insurance_display}")
        if 'behavioral_health_covered' in insurance_details:
            coverage_status = insurance_details['behavioral_health_covered']
            if coverage_status is True:
                print(f"  └─ Mental health coverage: ✓ Confirmed")
            elif coverage_status is False:
                print(f"  └─ Mental health coverage: ⚠️  Not covered - searching for alternatives")
            else:
                print(f"  └─ Mental health coverage: ? Unknown - will verify with facilities")
    else:
        print(f"✓ Insurance: {insurance_status or 'Not specified'}")
    
    print("─"*70)
    
    if is_crisis:
        print(f"\n🚢 Harbor: {user_name}, I'm finding crisis and mental health resources")
        print(f"          in {city} that can help you right now...\n")
    else:
        print(f"\n🚢 Harbor: Thank you, {user_name}. I'm searching for the best resources")
        print(f"          to support you in {city}...\n")
    
    # Step 5: Route using group2_router
    print("\n" + "┌" + "─"*68 + "┐")
    print("│" + " ⚙️  Step 3: Routing to Appropriate Services ".center(68) + "│")
    print("└" + "─"*68 + "┘\n")
    
    from data_adapter import adapt_llm_output
    normalized_classification = adapt_llm_output(classification)
    
    is_ours, routing_decision = handle_group2_input(normalized_classification)
    print(f"✓ {routing_decision['message']}\n")
    
    # Step 6: Handle based on routing decision
    if not is_ours:
        # Not our category - hand off
        print(f"→ This request should be handled by {routing_decision['branch']}")
        print("→ Passing to appropriate team...\n")
        return {
            'status': 'handed_off',
            'branch': routing_decision['branch'],
            'classification': normalized_classification
        }
    
    # Step 7: Normalize location from classification
    location = normalized_classification.get('location', {}) or {}
    insurance = normalized_classification.get('insurance', {}) or {}
    
    # Normalize location data (city capitalization + state abbreviation)
    if location.get('city') or location.get('state'):
        raw_city = location.get('city', '')
        raw_state = location.get('state', '')
        # Parse and normalize using our helper
        parsed_city, parsed_state = parse_location_input(f"{raw_city} {raw_state}")
        if parsed_city:
            location['city'] = parsed_city
        if parsed_state:
            location['state'] = parsed_state
    
    additional_info = {'location': location, 'insurance': insurance}

    # Always show both output formats (no need to ask user)
    additional_info['output_format'] = 'both'  # Signal to show both formats
    
    # Step 8: Match facilities
    print("\n" + "┌" + "─"*68 + "┐")
    print("│" + " 🔍 Step 4: Searching for Matching Facilities ".center(68) + "│")
    print("└" + "─"*68 + "┘\n")
    
    facilities = call_facility_matcher(normalized_classification, additional_info)
    
    # Step 9: Display results
    print("\n" + "═"*70)
    print("  ✅ SEARCH COMPLETE".center(70))
    print("═"*70)
    
    if not facilities:
        print("\nNote: Facility matching is still being refined.")
    
    # Enhanced thank you message based on category
    print("\n" + "─"*70)
    print(f"🚢 Harbor: {user_name}, thank you for trusting me with this.")
    
    # Personalized encouragement based on category
    category_lower = category.lower() if category else ""
    if 'anxiety' in category_lower:
        print("          Managing anxiety takes courage, and you've taken an important")
        print("          first step today. These facilities can provide the support")
        print("          you deserve.")
    elif 'depression' in category_lower or 'mood' in category_lower:
        print("          Seeking help for depression is a sign of strength, not weakness.")
        print("          These providers understand what you're going through and can")
        print("          help you find your way forward.")
    elif 'substance' in category_lower or 'addiction' in category_lower:
        print("          Recovery is possible, and you don't have to do this alone.")
        print("          These facilities specialize in supporting people on their")
        print("          journey to wellness.")
    else:
        print("          Taking this step to find support shows real strength.")
        print("          These providers are here to help you on your journey.")
    
    print("\n          💙 Remember: Healing isn't linear, and it's okay to ask for help.")
    print("          📞 If you need to talk to someone right away, the resources")
    print("          below are available 24/7.")
    print("─"*70)
    
    # Display emergency resources reminder
    print("\n" + "═"*70)
    print("  📋 ADDITIONAL RESOURCES & 24/7 CRISIS SUPPORT".center(70))
    print("═"*70)
    print("📞 Call/Text 988 (Suicide & Crisis Lifeline)")
    print("💬 Text HOME to 741741 (Crisis Text Line)")
    print("🏥 Call 1-800-662-4357 (SAMHSA National Helpline)")
    print("💙 Visit www.TheAdamProject.org (1,300+ free providers nationwide)")
    print("═"*70 + "\n")
    
    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Feature 12 - Comprehensive Online Resources
    # ═══════════════════════════════════════════════════════════════
    # Display comprehensive online mental health resources with direct website links
    print(display_online_resources())
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Feature 11 - Follow-up Support Reminders
    # ═══════════════════════════════════════════════════════════════
    # Provide severity-based follow-up recommendations and next steps
    crisis_severity = empathy_result.get('severity') if empathy_result.get('is_crisis') else None
    has_user_insurance = additional_info.get('insurance', {}).get('has_insurance', False)
    
    print(display_followup_support(user_name, severity=crisis_severity, has_insurance=has_user_insurance))
    print()
    
    return {
        'status': 'success',
        'classification': normalized_classification,
        'additional_info': additional_info,
        'facilities': facilities,
        'turn_count': turn_count
    }


# =====================================================
# Main Entry Point
# =====================================================

def main():
    """Main entry point for the pipeline"""
    
    try:
        result = run_pipeline()
        
        print("\n[Pipeline execution completed successfully]")
        print(f"Status: {result['status']}")
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        print("Exiting...")
    except Exception as e:
        print(f"\n\nERROR: Pipeline failed")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
