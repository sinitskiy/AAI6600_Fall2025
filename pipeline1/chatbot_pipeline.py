#!/usr/bin/env python3
"""
Mental Health Chatbot Pipeline - Complete Version with Category Taxonomy

Based on 80 College Student Scenarios document.
Includes detailed category classification and routing.

Author: Subgroup B (Michael & Radhika) - Enhanced Version
Date: November 2025
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Add paths for imports
current_file = Path(__file__).resolve()
root_dir = current_file.parent
p1_dir = root_dir / "p1"
integrated_dir = root_dir / "integrated"

sys.path.insert(0, str(p1_dir))
sys.path.insert(0, str(integrated_dir))

# Import existing modules
from routing.classification_router import handle_group2_input
import pandas as pd
import numpy as np
import re
import json
import requests
import google.generativeai as genai
from openai import OpenAI

# Global variable for outputting JSON data (for testing/debugging)
OUTPUT_JSON_DATA = False

# =====================================================
# PRIMARY CATEGORY DEFINITIONS (Based on 80 Scenarios)
# =====================================================

PRIMARY_CATEGORIES = {
    'CRISIS': {
        'name': 'Crisis/Emergency',
        'priority': 1,
        'description': 'Immediate danger, suicidal ideation, acute crisis',
        'color': '🔴'
    },
    'TRAUMA': {
        'name': 'Trauma & PTSD',
        'priority': 2,
        'description': 'Trauma, PTSD, acute stress reactions, violence-related',
        'color': '🟠'
    },
    'MOOD': {
        'name': 'Depression & Mood Disorders',
        'priority': 3,
        'description': 'Depression, mood swings, emotional regulation',
        'color': '🟡'
    },
    'ANXIETY': {
        'name': 'Anxiety Disorders',
        'priority': 4,
        'description': 'Various anxiety types, panic, phobias',
        'color': '🟡'
    },
    'SUBSTANCE': {
        'name': 'Substance Use & Addiction',
        'priority': 5,
        'description': 'Substance misuse, behavioral addiction, relapse',
        'color': '🟠'
    },
    'EATING': {
        'name': 'Eating & Body Image',
        'priority': 6,
        'description': 'Disordered eating, body image distress',
        'color': '🟡'
    },
    'ADJUSTMENT': {
        'name': 'Adjustment & Transition',
        'priority': 7,
        'description': 'Life transitions, adjustment disorders, culture shock',
        'color': '🟢'
    },
    'STRESS': {
        'name': 'Stress & Burnout',
        'priority': 8,
        'description': 'Academic stress, burnout, overload',
        'color': '🟢'
    },
    'GRIEF': {
        'name': 'Grief & Loss',
        'priority': 9,
        'description': 'Bereavement, loss, major life changes',
        'color': '🟡'
    },
    'IDENTITY': {
        'name': 'Identity & Belonging',
        'priority': 10,
        'description': 'LGBTQ+, cultural identity, imposter syndrome, belonging',
        'color': '🟢'
    },
    'RELATIONSHIP': {
        'name': 'Relationship & Interpersonal',
        'priority': 11,
        'description': 'Relationship issues, family stress, interpersonal conflict',
        'color': '🟢'
    },
    'ACADEMIC': {
        'name': 'Academic Performance',
        'priority': 12,
        'description': 'Academic struggles, executive dysfunction, career concerns',
        'color': '🟢'
    },
    'RESOURCE': {
        'name': 'Resource & Access Barriers',
        'priority': 13,
        'description': 'Financial stress, housing, accessibility issues',
        'color': '🟢'
    }
}

# =====================================================
# DETAILED MENTAL HEALTH TAXONOMY
# Maps specific conditions to categories with keywords
# =====================================================

MENTAL_HEALTH_TAXONOMY = {
    # ═══════════════════════════════════════════════════════════════
    # CRISIS (Priority 1) - Check FIRST
    # ═══════════════════════════════════════════════════════════════
    'crisis': {
        'primary': 'CRISIS',
        'subcategories': ['suicidal_ideation', 'self_harm', 'acute_crisis'],
        'keywords': [
            'suicide', 'suicidal', 'kill myself', 'end my life', 'want to die',
            'hurt myself', 'self-harm', 'self harm', 'cutting', 'overdose',
            'no reason to live', 'better off dead', 'ending it', "can't go on",
            'cant go on', 'crisis', 'emergency', 'immediate danger', 'not safe',
            'take my life', 'end it all', 'no point living', 'wish i was dead'
        ],
        'keywords_es': [
            'suicidio', 'suicida', 'matarme', 'terminar con mi vida', 'quiero morir',
            'hacerme daño', 'autolesión', 'cortarme', 'sobredosis', 'crisis',
            'emergencia', 'no quiero vivir', 'mejor muerto'
        ],
        'keywords_zh': [
            '自杀', '自殺', '想死', '不想活', '结束生命', '結束生命',
            '自残', '自殘', '割腕', '危机', '危機', '紧急', '緊急'
        ],
        'assistance_type': ['Crisis line', 'Psychiatrist', 'Emergency services'],
        'scenarios': [29]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # TRAUMA & PTSD (Priority 2)
    # ═══════════════════════════════════════════════════════════════
    'trauma_ptsd': {
        'primary': 'TRAUMA',
        'subcategories': ['ptsd', 'acute_stress', 'secondary_trauma', 'ipv'],
        'keywords': [
            'trauma', 'ptsd', 'traumatic', 'flashback', 'flashbacks', 'nightmare',
            'nightmares', 'assault', 'sexual assault', 'violence', 'abuse',
            'attacked', 'witnessed', 'accident', 'disaster', 'shooting', 'death',
            'rape', 'molested', 'abused', 'beaten', 'traumatized'
        ],
        'assistance_type': ['Trauma counseling', 'Crisis services', 'Group therapy'],
        'scenarios': [9, 28, 34, 60, 70, 76]
    },
    'racial_discrimination_trauma': {
        'primary': 'TRAUMA',
        'subcategories': ['racial_stress', 'microaggressions', 'discrimination'],
        'keywords': [
            'discrimination', 'racist', 'racism', 'microaggression', 'microaggressions',
            'racial', 'prejudice', 'bias', 'hate', 'targeted', 'slur', 'n-word',
            'hate crime', 'racial profiling', 'stereotyped'
        ],
        'assistance_type': ['Cultural counseling', 'Support group'],
        'scenarios': [9, 50]
    },
    'secondary_trauma': {
        'primary': 'TRAUMA',
        'subcategories': ['vicarious_trauma', 'witness_trauma'],
        'keywords': [
            'witnessed', 'saw someone', 'friend overdose', 'friend hurt',
            'someone died', 'watched', 'helpless watching', 'saw accident',
            'friend suicide', 'roommate hurt'
        ],
        'assistance_type': ['Crisis counseling', 'Trauma therapy'],
        'scenarios': [34, 60, 76]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # DEPRESSION & MOOD (Priority 3)
    # ═══════════════════════════════════════════════════════════════
    'depression': {
        'primary': 'MOOD',
        'subcategories': ['major_depression', 'persistent_depression', 'low_mood'],
        'keywords': [
            'depressed', 'depression', 'hopeless', 'worthless', 'empty',
            'no motivation', 'cant get out of bed', "can't function",
            'nothing matters', 'numb', 'sad all the time', 'crying',
            'no energy', 'tired all the time', 'dont care anymore',
            'lost interest', 'no joy', 'miserable', 'despair'
        ],
        'keywords_es': [
            'deprimido', 'deprimida', 'depresión', 'sin esperanza', 'vacío',
            'sin motivación', 'triste', 'llorando', 'sin energía', 'miserable'
        ],
        'keywords_zh': [
            '抑郁', '抑鬱', '绝望', '絕望', '空虚', '空虛', '没有动力',
            '沒有動力', '悲伤', '悲傷', '痛苦', '难过', '難過'
        ],
        'assistance_type': ['Counseling', 'Psychiatrist'],
        'scenarios': [3]
    },
    'mood_disorder': {
        'primary': 'MOOD',
        'subcategories': ['bipolar', 'mood_swings', 'emotional_dysregulation'],
        'keywords': [
            'mood swings', 'bipolar', 'manic', 'mania', 'up and down',
            'irritable', 'cant control emotions', 'angry outbursts', 'rage',
            'emotional', 'unstable mood', 'explosive', 'flying off handle',
            'highs and lows', 'rapid mood changes'
        ],
        'assistance_type': ['Psychiatrist', 'Counseling'],
        'scenarios': [24, 59]
    },
    'emotional_blunting': {
        'primary': 'MOOD',
        'subcategories': ['numbness', 'disconnection'],
        'keywords': [
            'numb', 'cant feel', 'emotionless', 'disconnected', 'detached',
            'dont feel anything', 'empty inside', 'flat', 'no emotions',
            'cant cry', 'robot', 'going through motions'
        ],
        'assistance_type': ['Counseling', 'Psychiatrist'],
        'scenarios': [39]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # ANXIETY DISORDERS (Priority 4)
    # ═══════════════════════════════════════════════════════════════
    'general_anxiety': {
        'primary': 'ANXIETY',
        'subcategories': ['gad', 'worry', 'nervousness'],
        'keywords': [
            'anxious', 'anxiety', 'worried', 'worrying', 'nervous', 'on edge',
            'cant relax', 'racing thoughts', 'restless', 'tense', 'keyed up',
            'fear', 'dread', 'panic', 'panicking', 'overwhelmed', 'stressed out',
            'heart racing', 'cant breathe', 'chest tight', 'shaking'
        ],
        'keywords_es': [
            'ansioso', 'ansiosa', 'ansiedad', 'preocupado', 'preocupada',
            'nervioso', 'nerviosa', 'pánico', 'miedo', 'tenso', 'tensa'
        ],
        'keywords_zh': [
            '焦虑', '焦慮', '担心', '擔心', '紧张', '緊張', '恐慌',
            '害怕', '不安', '心慌'
        ],
        'assistance_type': ['Counseling', 'Self-help'],
        'scenarios': [12]
    },
    'test_anxiety': {
        'primary': 'ANXIETY',
        'subcategories': ['exam_stress', 'performance_anxiety'],
        'keywords': [
            'test anxiety', 'exam stress', 'panic during test', 'exam anxiety',
            'blank out', 'fail exam', 'cant concentrate exam', 'test panic',
            'nervous about grades', 'scared of failing', 'bombed test',
            'mind goes blank', 'freeze during exam', 'finals stress'
        ],
        'assistance_type': ['Counseling', 'Self-help', 'Academic support'],
        'scenarios': [2, 11]
    },
    'social_anxiety': {
        'primary': 'ANXIETY',
        'subcategories': ['social_phobia', 'public_speaking', 'group_anxiety'],
        'keywords': [
            'social anxiety', 'afraid of people', 'public speaking', 'shy',
            'presentation anxiety', 'group work scared', 'avoid social',
            'embarrassed', 'judged', 'awkward', 'social situations',
            'talking to people', 'meeting new people', 'parties scary',
            'hate speaking up', 'class participation'
        ],
        'assistance_type': ['Skills group', 'Counseling'],
        'scenarios': [38, 47]
    },
    'health_anxiety': {
        'primary': 'ANXIETY',
        'subcategories': ['hypochondria', 'medical_worry'],
        'keywords': [
            'health anxiety', 'worried sick', 'afraid of illness', 'sick',
            'hypochondriac', 'checking symptoms', 'fear of disease',
            'vaccine anxiety', 'medical fear', 'something wrong with me',
            'cancer scared', 'googling symptoms', 'health scare'
        ],
        'assistance_type': ['Campus health', 'Counseling'],
        'scenarios': [40, 53]
    },
    'ocd_anxiety': {
        'primary': 'ANXIETY',
        'subcategories': ['ocd', 'intrusive_thoughts', 'compulsions'],
        'keywords': [
            'ocd', 'obsessive', 'compulsive', 'intrusive thoughts',
            'cant stop thinking', 'rituals', 'checking', 'contamination',
            'unwanted thoughts', 'have to do', 'counting', 'repeating',
            'stuck in my head', 'obsessing', 'compelled to'
        ],
        'assistance_type': ['Psychiatrist', 'Counseling'],
        'scenarios': [37]
    },
    'safety_anxiety': {
        'primary': 'ANXIETY',
        'subcategories': ['campus_safety', 'housing_safety'],
        'keywords': [
            'unsafe', 'not safe', 'scared in dorm', 'afraid at night',
            'security', 'danger', 'threatened', 'stalked', 'stalker',
            'followed', 'harassed', 'scared to walk', 'campus unsafe'
        ],
        'assistance_type': ['Counseling', 'Student services', 'Campus security'],
        'scenarios': [18]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # SUBSTANCE USE & ADDICTION (Priority 5)
    # ═══════════════════════════════════════════════════════════════
    'substance_misuse': {
        'primary': 'SUBSTANCE',
        'subcategories': ['alcohol', 'drugs', 'medication_misuse'],
        'keywords': [
            'drinking', 'alcohol', 'drunk', 'drugs', 'marijuana', 'weed',
            'pills', 'cocaine', 'adderall', 'substance', 'high', 'stoned',
            'using', 'getting wasted', 'blacking out', 'blackout',
            'party too much', 'cant stop drinking', 'need to drink',
            'smoking weed', 'xanax', 'opioids', 'heroin', 'fentanyl'
        ],
        'keywords_es': [
            'alcohol', 'drogas', 'marihuana', 'borracho', 'borracha',
            'adicción', 'adiccion', 'bebiendo', 'usando drogas'
        ],
        'keywords_zh': [
            '酗酒', '毒品', '上瘾', '上癮', '喝酒', '吸毒', '大麻'
        ],
        'assistance_type': ['Counseling', 'Psychiatrist', 'AA/NA'],
        'scenarios': [8]
    },
    'substance_relapse': {
        'primary': 'SUBSTANCE',
        'subcategories': ['relapse', 'recovery_struggle'],
        'keywords': [
            'relapse', 'relapsed', 'started using again', 'fell off wagon',
            'recovery', 'sober', 'clean', 'struggled', 'addiction',
            'back to drinking', 'using again', 'slipped up', 'broke sobriety'
        ],
        'assistance_type': ['Psychiatrist', 'Specialist', 'Support group'],
        'scenarios': [51]
    },
    'behavioral_addiction': {
        'primary': 'SUBSTANCE',
        'subcategories': ['gaming', 'internet', 'gambling', 'shopping'],
        'keywords': [
            'gaming addiction', 'video game', 'cant stop playing', 'gaming',
            'internet addiction', 'social media addiction', 'phone addiction',
            'gambling', 'porn', 'pornography', 'shopping addiction',
            'addicted to phone', 'screen time', 'cant put down phone',
            'playing all night', 'lost track of time gaming'
        ],
        'assistance_type': ['Self-care', 'Counseling'],
        'scenarios': [32]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # EATING & BODY IMAGE (Priority 6)
    # ═══════════════════════════════════════════════════════════════
    'disordered_eating': {
        'primary': 'EATING',
        'subcategories': ['anorexia', 'bulimia', 'binge_eating', 'restriction'],
        'keywords': [
            'eating disorder', 'anorexia', 'bulimia', 'binge eating', 'bingeing',
            'purging', 'throwing up', 'not eating', 'restricting', 'starving',
            'calorie counting', 'food fear', 'overeating', 'cant stop eating',
            'skipping meals', 'laxatives', 'diet obsessed', 'food rules'
        ],
        'assistance_type': ['Counseling', 'Nutritionist', 'Psychiatrist'],
        'scenarios': [17]
    },
    'body_image': {
        'primary': 'EATING',
        'subcategories': ['body_dysmorphia', 'weight_concerns'],
        'keywords': [
            'body image', 'hate my body', 'fat', 'ugly', 'weight', 'thin',
            'body shaming', 'appearance', 'mirror', 'disgusted with myself',
            'look horrible', 'too fat', 'too skinny', 'body dysmorphia',
            'hate how i look', 'wish i looked different'
        ],
        'assistance_type': ['Counseling'],
        'scenarios': [41]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # ADJUSTMENT & TRANSITION (Priority 7)
    # ═══════════════════════════════════════════════════════════════
    'adjustment_disorder': {
        'primary': 'ADJUSTMENT',
        'subcategories': ['life_change', 'new_situation'],
        'keywords': [
            'adjusting', 'new school', 'new environment', 'change', 'changes',
            'hard to adapt', 'different', 'struggling to fit', 'new place',
            'everything is new', 'dont know anyone', 'starting fresh'
        ],
        'assistance_type': ['Counseling', 'Peer support'],
        'scenarios': [4, 13, 48]
    },
    'homesickness': {
        'primary': 'ADJUSTMENT',
        'subcategories': ['loneliness', 'missing_home'],
        'keywords': [
            'homesick', 'miss home', 'miss family', 'miss my mom', 'miss my dad',
            'lonely', 'alone', 'no friends', 'isolated', 'far from home',
            'want to go home', 'miss my friends', 'nobody here', 'all alone'
        ],
        'assistance_type': ['Self-care', 'Peer group', 'Counseling'],
        'scenarios': [1, 22]
    },
    'culture_shock': {
        'primary': 'ADJUSTMENT',
        'subcategories': ['international', 'cultural_adjustment', 'reverse_culture_shock'],
        'keywords': [
            'culture shock', 'international student', 'language barrier',
            'different culture', 'miss my country', 'cultural food', 'accent',
            'traditions', 'reverse culture shock', 'study abroad', 'foreign',
            'dont understand customs', 'feel like outsider', 'language hard'
        ],
        'assistance_type': ['Counseling', 'Cultural center', 'Peer support'],
        'scenarios': [4, 45, 57]
    },
    'transition_stress': {
        'primary': 'ADJUSTMENT',
        'subcategories': ['transfer', 'returning', 'medical_leave_return'],
        'keywords': [
            'transfer student', 'new college', 'returning', 'medical leave',
            'coming back', 'starting over', 're-adjusting', 'readjusting',
            'back to school', 'transferred', 'switched schools'
        ],
        'assistance_type': ['Peer mentor', 'Counseling', 'Academic support'],
        'scenarios': [21, 48, 71]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # STRESS & BURNOUT (Priority 8)
    # ═══════════════════════════════════════════════════════════════
    'academic_stress': {
        'primary': 'STRESS',
        'subcategories': ['coursework', 'thesis', 'grades'],
        'keywords': [
            'academic stress', 'too much homework', 'overwhelmed with work',
            'thesis stress', 'dissertation', 'grades dropping', 'gpa',
            'failing class', 'cant keep up', 'assignments piling up',
            'behind on work', 'due dates', 'deadlines', 'midterms', 'finals'
        ],
        'assistance_type': ['Counseling', 'Academic advisor'],
        'scenarios': [26, 35, 52, 80]
    },
    'burnout': {
        'primary': 'STRESS',
        'subcategories': ['exhaustion', 'overwork', 'perfectionism'],
        'keywords': [
            'burnout', 'burned out', 'burnt out', 'exhausted', 'overwhelmed',
            'too much', 'cant do this anymore', 'drained', 'running on empty',
            'perfectionism', 'never good enough', 'pushing too hard',
            'no breaks', 'always working', 'never stops'
        ],
        'assistance_type': ['Counseling', 'Self-care'],
        'scenarios': [6, 23, 68]
    },
    'work_life_balance': {
        'primary': 'STRESS',
        'subcategories': ['overload', 'caregiving', 'multiple_roles'],
        'keywords': [
            'work and school', 'job and classes', 'caregiving', 'caregiver',
            'taking care of family', 'no time', 'balancing', 'juggling',
            'cant do everything', 'too many responsibilities', 'spread thin',
            'working too much', 'no personal time'
        ],
        'assistance_type': ['Counseling', 'Case manager'],
        'scenarios': [6, 77]
    },
    'competitive_stress': {
        'primary': 'STRESS',
        'subcategories': ['peer_competition', 'pre_med', 'high_achiever'],
        'keywords': [
            'competitive', 'everyone is better', 'comparing myself', 'compared',
            'pre-med stress', 'premed', 'not good enough', 'behind peers',
            'imposter', 'dont belong', 'peers smarter', 'curve', 'rankings'
        ],
        'assistance_type': ['Peer support', 'Counseling', 'Mentoring'],
        'scenarios': [73, 78]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # GRIEF & LOSS (Priority 9)
    # ═══════════════════════════════════════════════════════════════
    'grief_bereavement': {
        'primary': 'GRIEF',
        'subcategories': ['death', 'loss', 'mourning'],
        'keywords': [
            'grief', 'grieving', 'death', 'died', 'lost someone', 'passed away',
            'funeral', 'mourning', 'miss them', 'bereavement', 'gone forever',
            'cant believe theyre gone', 'lost my', 'death of', 'killed'
        ],
        'assistance_type': ['Counseling', 'Grief support group'],
        'scenarios': [10, 44]
    },
    'family_illness': {
        'primary': 'GRIEF',
        'subcategories': ['sick_family', 'caregiver_stress'],
        'keywords': [
            'family sick', 'parent cancer', 'family member ill', 'mom sick',
            'dad sick', 'hospital', 'diagnosis', 'terminal', 'dying',
            'worried about family', 'family health', 'serious illness'
        ],
        'assistance_type': ['Counseling', 'Support group'],
        'scenarios': [42]
    },
    'professional_loss': {
        'primary': 'GRIEF',
        'subcategories': ['mentor_loss', 'advisor_change'],
        'keywords': [
            'advisor left', 'mentor gone', 'professor left', 'lost my mentor',
            'abandoned', 'no guidance', 'advisor quit', 'mentor died'
        ],
        'assistance_type': ['Counseling', 'Mentoring'],
        'scenarios': [69]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # IDENTITY & BELONGING (Priority 10)
    # ═══════════════════════════════════════════════════════════════
    'lgbtq_stress': {
        'primary': 'IDENTITY',
        'subcategories': ['coming_out', 'gender_identity', 'sexual_orientation'],
        'keywords': [
            'lgbtq', 'lgbt', 'gay', 'lesbian', 'bisexual', 'transgender', 'trans',
            'queer', 'coming out', 'outed', 'gender identity', 'sexuality',
            'non-binary', 'nonbinary', 'questioning', 'closeted', 'pronouns',
            'transition', 'dysphoria', 'homophobic', 'transphobic'
        ],
        'assistance_type': ['Counseling', 'LGBTQ+ resource center', 'Group therapy'],
        'scenarios': [27, 55]
    },
    'cultural_identity': {
        'primary': 'IDENTITY',
        'subcategories': ['ethnic_identity', 'bicultural', 'heritage'],
        'keywords': [
            'cultural identity', 'ethnic', 'heritage', 'between cultures',
            'dont fit in', 'too american', 'not american enough', 'bicultural',
            'different background', 'minority', 'ethnic identity', 'race',
            'where do i belong', 'identity crisis'
        ],
        'assistance_type': ['Counseling', 'Cultural center'],
        'scenarios': [19, 45, 79]
    },
    'imposter_syndrome': {
        'primary': 'IDENTITY',
        'subcategories': ['self_doubt', 'first_gen'],
        'keywords': [
            'imposter syndrome', 'imposter', 'fraud', 'dont deserve',
            'not smart enough', 'first generation', 'first-gen', 'first gen',
            'dont belong here', 'will find out', 'not qualified', 'fake',
            'everyone else knows', 'out of my league'
        ],
        'assistance_type': ['Mentoring', 'Counseling', 'Peer support'],
        'scenarios': [16]
    },
    'belonging_anxiety': {
        'primary': 'IDENTITY',
        'subcategories': ['fitting_in', 'marginalized'],
        'keywords': [
            'dont belong', 'outsider', 'different', 'marginalized',
            'excluded', 'invisible', 'not welcome', 'campus climate',
            'dont fit', 'left out', 'no one like me', 'underrepresented'
        ],
        'assistance_type': ['Multicultural center', 'Counseling', 'Peer support'],
        'scenarios': [62, 78, 79]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # RELATIONSHIP & INTERPERSONAL (Priority 11)
    # ═══════════════════════════════════════════════════════════════
    'relationship_stress': {
        'primary': 'RELATIONSHIP',
        'subcategories': ['breakup', 'romantic', 'dating'],
        'keywords': [
            'breakup', 'broke up', 'relationship', 'boyfriend', 'girlfriend',
            'partner', 'dating', 'heartbreak', 'heartbroken', 'cheated',
            'trust issues', 'ex', 'dumped', 'rejected', 'love problems'
        ],
        'assistance_type': ['Self-care', 'Peer support', 'Counseling'],
        'scenarios': [5]
    },
    'family_stress': {
        'primary': 'RELATIONSHIP',
        'subcategories': ['parent_conflict', 'divorce', 'family_expectations'],
        'keywords': [
            'family', 'parents', 'divorce', 'family pressure', 'mom and dad',
            'family expectations', 'disappointing family', 'family conflict',
            'home problems', 'toxic family', 'family fighting', 'siblings',
            'family drama', 'parents fighting', 'parents divorced'
        ],
        'assistance_type': ['Self-care', 'Counseling'],
        'scenarios': [11, 33, 42]
    },
    'interpersonal_conflict': {
        'primary': 'RELATIONSHIP',
        'subcategories': ['roommate', 'peer_conflict', 'group_conflict'],
        'keywords': [
            'roommate', 'roommate problems', 'conflict', 'fight with',
            'argument', 'arguing', 'group project', 'teammate', 'resentment',
            'cant stand', 'hate my roommate', 'suitemate', 'living situation'
        ],
        'assistance_type': ['Mediation', 'Peer support'],
        'scenarios': [7, 54, 59]
    },
    'cyberbullying': {
        'primary': 'RELATIONSHIP',
        'subcategories': ['online_harassment', 'social_media_bullying'],
        'keywords': [
            'cyberbullying', 'cyberbullied', 'online harassment', 'social media',
            'trolling', 'hate messages', 'doxxed', 'canceled', 'cancelled',
            'targeted online', 'mean comments', 'online hate', 'harassment'
        ],
        'assistance_type': ['Counseling', 'Online safety resources'],
        'scenarios': [74]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # ACADEMIC PERFORMANCE (Priority 12)
    # ═══════════════════════════════════════════════════════════════
    'executive_dysfunction': {
        'primary': 'ACADEMIC',
        'subcategories': ['adhd', 'procrastination', 'focus_issues'],
        'keywords': [
            'procrastination', 'procrastinating', 'cant focus', 'cant start',
            'adhd', 'add', 'attention', 'distracted', 'executive function',
            'cant plan', 'disorganized', 'time management', 'putting off',
            'last minute', 'cant concentrate', 'unfocused'
        ],
        'assistance_type': ['Academic coaching', 'Counseling'],
        'scenarios': [35]
    },
    'career_anxiety': {
        'primary': 'ACADEMIC',
        'subcategories': ['future_worry', 'job_market', 'wrong_major'],
        'keywords': [
            'career', 'job', 'future', 'what to do with life', 'after graduation',
            'wrong major', 'degree worthless', 'job market', 'unemployed',
            'interview anxiety', 'career path', 'no jobs', 'wasted degree',
            'dont know what to do', 'career change'
        ],
        'assistance_type': ['Career counseling', 'Advising'],
        'scenarios': [25, 75]
    },
    'academic_setback': {
        'primary': 'ACADEMIC',
        'subcategories': ['probation', 'failed_exam', 'suspension'],
        'keywords': [
            'academic probation', 'probation', 'failed', 'failing', 'suspension',
            'kicked out', 'academic integrity', 'cheating investigation',
            'licensure exam', 'board exam', 'academic dishonesty', 'expelled',
            'dismissed', 'academic warning'
        ],
        'assistance_type': ['Academic advising', 'Counseling'],
        'scenarios': [31, 52, 80]
    },
    
    # ═══════════════════════════════════════════════════════════════
    # RESOURCE & ACCESS BARRIERS (Priority 13)
    # ═══════════════════════════════════════════════════════════════
    'financial_stress': {
        'primary': 'RESOURCE',
        'subcategories': ['money_worry', 'food_insecurity', 'cant_afford'],
        'keywords': [
            'financial', 'money', 'broke', 'cant afford', 'debt', 'loans',
            'food insecurity', 'hungry', 'textbooks', 'tuition', 'rent',
            'working too much', 'bills', 'poor', 'no money', 'struggling financially'
        ],
        'assistance_type': ['Case manager', 'Financial aid', 'Mental health support'],
        'scenarios': [14, 36, 43, 58]
    },
    'housing_insecurity': {
        'primary': 'RESOURCE',
        'subcategories': ['homeless', 'eviction', 'unstable_housing'],
        'keywords': [
            'housing', 'homeless', 'evicted', 'eviction', 'no place to stay',
            'couch surfing', 'housing insecure', 'cant pay rent', 'kicked out',
            'nowhere to live', 'sleeping in car', 'shelter'
        ],
        'assistance_type': ['Crisis services', 'Case manager'],
        'scenarios': [63]
    },
    'accessibility_stress': {
        'primary': 'RESOURCE',
        'subcategories': ['disability', 'accommodations', 'disclosure'],
        'keywords': [
            'disability', 'accommodations', 'accessible', 'chronic illness',
            'chronic pain', 'disabled', 'disclosure', 'ada', 'iep',
            'cant get help', 'not supported', 'need accommodations',
            'accessibility', 'wheelchair', 'blind', 'deaf', 'learning disability'
        ],
        'assistance_type': ['Disability services', 'Mental health support'],
        'scenarios': [20, 49, 61]
    },
    'geographic_isolation': {
        'primary': 'RESOURCE',
        'subcategories': ['rural', 'remote', 'commuter'],
        'keywords': [
            'rural', 'remote', 'far from services', 'commuter', 'commuting',
            'no counseling available', 'limited resources', 'isolated campus',
            'no therapists nearby', 'long drive', 'no public transit'
        ],
        'assistance_type': ['Virtual counseling', 'Telehealth'],
        'scenarios': [46, 62]
    },
    'military_connected': {
        'primary': 'RESOURCE',
        'subcategories': ['veteran', 'deployment', 'military_family'],
        'keywords': [
            'military', 'veteran', 'deployed', 'deployment', 'service member',
            'army', 'navy', 'marines', 'air force', 'national guard',
            'military family', 'va', 'gi bill', 'combat', 'ptsd military'
        ],
        'assistance_type': ['Military student support', 'VA services'],
        'scenarios': [64]
    },
    'parenting_student': {
        'primary': 'RESOURCE',
        'subcategories': ['student_parent', 'pregnancy', 'childcare'],
        'keywords': [
            'pregnant', 'pregnancy', 'baby', 'child', 'parenting', 'mother',
            'father', 'childcare', 'daycare', 'single parent', 'kids',
            'student parent', 'breastfeeding', 'maternity'
        ],
        'assistance_type': ['Student services', 'Parenting support'],
        'scenarios': [65]
    }
}

# =====================================================
# ASSISTANCE TYPE MAPPING
# =====================================================

ASSISTANCE_TYPE_MAPPING = {
    'Crisis line': {
        'resources': ['988 Suicide & Crisis Lifeline', 'Crisis Text Line (741741)', '911'],
        'urgency': 'immediate',
        'categories': ['CRISIS']
    },
    'Psychiatrist': {
        'resources': ['Campus psychiatry', 'Community mental health center', 'Telehealth psychiatry'],
        'urgency': 'high',
        'categories': ['CRISIS', 'MOOD', 'ANXIETY', 'SUBSTANCE']
    },
    'Counseling': {
        'resources': ['Campus counseling center', 'Community therapist', 'TheAdamProject.org'],
        'urgency': 'standard',
        'categories': ['MOOD', 'ANXIETY', 'TRAUMA', 'ADJUSTMENT', 'STRESS', 'GRIEF', 'IDENTITY', 'RELATIONSHIP', 'ACADEMIC', 'RESOURCE']
    },
    'Trauma counseling': {
        'resources': ['EMDR specialist', 'Trauma-informed therapist', 'Rape crisis center'],
        'urgency': 'high',
        'categories': ['TRAUMA']
    },
    'Self-care': {
        'resources': ['Wellness apps (Headspace, Calm)', 'Self-help guides', 'Campus wellness'],
        'urgency': 'low',
        'categories': ['ADJUSTMENT', 'STRESS', 'RELATIONSHIP']
    },
    'Peer support': {
        'resources': ['Support groups', 'Peer mentoring', 'NAMI groups'],
        'urgency': 'standard',
        'categories': ['ADJUSTMENT', 'IDENTITY', 'GRIEF', 'STRESS']
    },
    'Skills group': {
        'resources': ['DBT group', 'CBT group', 'Social skills training'],
        'urgency': 'standard',
        'categories': ['ANXIETY', 'MOOD']
    },
    'Academic support': {
        'resources': ['Academic advisor', 'Tutoring center', 'Study skills workshop'],
        'urgency': 'standard',
        'categories': ['ACADEMIC', 'STRESS']
    },
    'Case manager': {
        'resources': ['Campus case management', 'Social services', 'Basic needs hub'],
        'urgency': 'standard',
        'categories': ['RESOURCE']
    },
    'Cultural center': {
        'resources': ['Multicultural center', 'International student office', 'Identity-based orgs'],
        'urgency': 'standard',
        'categories': ['IDENTITY', 'ADJUSTMENT']
    },
    'LGBTQ+ resource': {
        'resources': ['LGBTQ+ center', 'Pride alliance', 'Trevor Project'],
        'urgency': 'standard',
        'categories': ['IDENTITY']
    },
    'Disability services': {
        'resources': ['Office of disability services', 'Accessibility coordinator'],
        'urgency': 'standard',
        'categories': ['RESOURCE']
    }
}

# =====================================================
# CRISIS DETECTION SYSTEM
# =====================================================

CRISIS_MODEL = None
CRISIS_EMBEDDINGS = None
HF_CLIENT = None

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
    """Initialize the sentence transformer model for crisis detection."""
    global CRISIS_MODEL, CRISIS_EMBEDDINGS
    
    if CRISIS_MODEL is not None:
        return
    
    try:
        from sentence_transformers import SentenceTransformer
        print("[Initializing crisis detection system...]")
        CRISIS_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        CRISIS_EMBEDDINGS = CRISIS_MODEL.encode(CRISIS_REFERENCE_PHRASES)
        print("✓ Crisis detection ready\n")
    except ImportError:
        print("⚠️  Warning: sentence-transformers not installed.")
        print("   Falling back to keyword + Gemini detection.\n")
        CRISIS_MODEL = False
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize crisis detection: {e}")
        CRISIS_MODEL = False


def detect_crisis_semantic(user_message, threshold=0.65):
    """Stage 2: Use semantic similarity with sentence transformers."""
    global CRISIS_MODEL, CRISIS_EMBEDDINGS
    
    if CRISIS_MODEL is None:
        initialize_crisis_detection()
    
    if CRISIS_MODEL is False:
        return {
            'is_crisis': None,
            'confidence': 0.0,
            'matched_phrase': None,
            'method': 'semantic_unavailable'
        }
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        user_embedding = CRISIS_MODEL.encode([user_message])[0]
        similarities = cosine_similarity([user_embedding], CRISIS_EMBEDDINGS)[0]
        
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
    """Stage 3: Use Gemini API for accurate crisis classification."""
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
}}"""
    
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    params = {"key": api_key}
    
    try:
        response = requests.post(endpoint, headers=headers, params=params, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        
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
    except Exception as e:
        pass
    
    return {
        'is_crisis': None,
        'confidence': 0.0,
        'crisis_type': 'unknown',
        'reasoning': 'Gemini API error',
        'method': 'gemini_error'
    }


def detect_crisis_hybrid(user_message):
    """Hybrid 3-stage crisis detection system."""
    message_lower = user_message.lower()
    
    # Stage 1: Fast keyword screening
    urgent_keywords = MENTAL_HEALTH_TAXONOMY.get('crisis', {}).get('keywords', [])
    has_urgent_keyword = any(kw in message_lower for kw in urgent_keywords)
    
    if not has_urgent_keyword:
        return {
            'is_crisis': False,
            'confidence': 0.95,
            'method': 'keyword_screening',
            'details': {'stage': 1, 'matched_keyword': None}
        }
    
    # Stage 2: Semantic similarity
    semantic_result = detect_crisis_semantic(user_message, threshold=0.65)
    
    if semantic_result['is_crisis'] is not None:
        if semantic_result['confidence'] >= 0.65:
            return {
                'is_crisis': semantic_result['is_crisis'],
                'confidence': semantic_result['confidence'],
                'method': 'semantic_trusted',
                'details': {'stage': 2, 'matched_phrase': semantic_result['matched_phrase']}
            }
    
    # Stage 3: Gemini confirmation
    print("   [Double-checking with AI for safety...]")
    gemini_result = detect_crisis_gemini(user_message)
    
    if gemini_result['is_crisis'] is not None:
        return {
            'is_crisis': gemini_result['is_crisis'],
            'confidence': gemini_result['confidence'],
            'method': 'gemini_confirmation',
            'details': {'stage': 3, 'crisis_type': gemini_result['crisis_type']}
        }
    
    # Fallback: treat as crisis if keywords present
    return {
        'is_crisis': True,
        'confidence': 0.70,
        'method': 'fallback_safe_default',
        'details': {'stage': 'fallback'}
    }


def assess_crisis_severity(user_message, crisis_result):
    """Assess the severity level of a crisis situation."""
    message_lower = user_message.lower()
    confidence = crisis_result.get('confidence', 0)
    
    immediate_keywords = [
        'right now', 'tonight', 'today', 'plan to', 'planning', 'going to',
        'about to', 'ready to', 'have a gun', 'have pills', 'overdose',
        'jump', 'hanging', 'cut my wrists'
    ]
    
    high_urgency_keywords = [
        'want to die', 'wish i was dead', 'better off dead', 'ending it',
        'kill myself', 'suicide', 'take my life', 'end my life'
    ]
    
    has_immediate = any(kw in message_lower for kw in immediate_keywords)
    has_high_urgency = any(kw in message_lower for kw in high_urgency_keywords)
    
    if has_immediate:
        return {
            'severity': 'immediate',
            'urgency_score': 10,
            'recommended_action': 'Call 911 immediately or go to nearest emergency room'
        }
    elif has_high_urgency and confidence > 0.8:
        return {
            'severity': 'high',
            'urgency_score': 8,
            'recommended_action': 'Call 988 (Suicide Prevention Lifeline) now - available 24/7'
        }
    elif has_high_urgency:
        return {
            'severity': 'high',
            'urgency_score': 7,
            'recommended_action': 'Call 988 or text HOME to 741741 for immediate support'
        }
    else:
        return {
            'severity': 'moderate',
            'urgency_score': 5,
            'recommended_action': 'Contact crisis support (988 or Crisis Text Line) soon'
        }


# =====================================================
# CATEGORY CLASSIFICATION SYSTEM
# =====================================================

def classify_mental_health_category(user_message, conversation_history=None):
    """
    Classify user's mental health concern into specific categories.
    Uses hybrid approach: keyword matching + Gemini fallback.
    """
    message_lower = user_message.lower()
    
    # Combine with conversation history if available
    if conversation_history:
        history_text = " ".join([
            msg.get('message', '') for msg in conversation_history 
            if msg.get('role') == 'USER'
        ])
        full_text = f"{history_text} {user_message}".lower()
    else:
        full_text = message_lower
    
    # Score all categories based on keyword matches
    category_scores = defaultdict(lambda: {'score': 0, 'matched_keywords': []})
    
    for cat_key, cat_details in MENTAL_HEALTH_TAXONOMY.items():
        for keyword in cat_details['keywords']:
            if keyword in full_text:
                category_scores[cat_key]['score'] += 1
                category_scores[cat_key]['matched_keywords'].append(keyword)
                category_scores[cat_key]['primary'] = cat_details['primary']
                category_scores[cat_key]['assistance'] = cat_details['assistance_type']
        
        # Check language-specific keywords
        for lang_key in ['keywords_es', 'keywords_zh']:
            if lang_key in cat_details:
                for keyword in cat_details[lang_key]:
                    if keyword in full_text or keyword in user_message:
                        category_scores[cat_key]['score'] += 1
                        category_scores[cat_key]['matched_keywords'].append(keyword)
                        category_scores[cat_key]['primary'] = cat_details['primary']
                        category_scores[cat_key]['assistance'] = cat_details['assistance_type']
    
    # Check for crisis indicators FIRST
    crisis_score = category_scores.get('crisis', {}).get('score', 0)
    if crisis_score > 0:
        return {
            'primary_category': 'CRISIS',
            'specific_category': 'crisis',
            'confidence': 0.95,
            'all_matches': [{'category': 'crisis', 'score': crisis_score}],
            'assistance_types': ['Crisis line', 'Psychiatrist', 'Emergency services'],
            'severity': 'crisis',
            'method': 'keyword',
            'matched_keywords': category_scores['crisis']['matched_keywords']
        }
    
    # Rank non-crisis categories
    if category_scores:
        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: (
                -x[1]['score'],
                PRIMARY_CATEGORIES.get(x[1].get('primary', 'RESOURCE'), {}).get('priority', 99)
            )
        )
        
        top_category = sorted_categories[0]
        cat_key = top_category[0]
        cat_data = top_category[1]
        
        score = cat_data['score']
        if score >= 3:
            confidence = 0.90
        elif score >= 2:
            confidence = 0.80
        else:
            confidence = 0.65
        
        primary = cat_data.get('primary', 'RESOURCE')
        priority = PRIMARY_CATEGORIES.get(primary, {}).get('priority', 99)
        
        if priority <= 2:
            severity = 'high'
        elif priority <= 6:
            severity = 'moderate'
        else:
            severity = 'low'
        
        all_matches = [
            {'category': k, 'score': v['score'], 'primary': v.get('primary')}
            for k, v in sorted_categories[:5]
        ]
        
        return {
            'primary_category': primary,
            'specific_category': cat_key,
            'confidence': confidence,
            'all_matches': all_matches,
            'assistance_types': cat_data.get('assistance', ['Counseling']),
            'severity': severity,
            'method': 'keyword',
            'matched_keywords': cat_data['matched_keywords']
        }
    
    # No keyword matches - use Gemini
    return classify_with_gemini_detailed(user_message)


def classify_with_gemini_detailed(user_message):
    """Use Gemini API for detailed mental health classification."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path) as f:
            config = json.load(f)
        api_key = config.get("GEMINI_API_KEY")
    except Exception:
        return _fallback_classification()
    
    if not api_key:
        return _fallback_classification()
    
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    category_list = "\n".join([
        f"- {key}: {PRIMARY_CATEGORIES[key]['description']}"
        for key in PRIMARY_CATEGORIES.keys()
    ])
    
    prompt = f"""Classify this mental health concern into the most appropriate category.

User's message: "{user_message}"

Available PRIMARY categories:
{category_list}

Respond with ONLY a JSON object (no markdown):
{{
    "primary_category": "CATEGORY_KEY",
    "specific_concern": "brief description",
    "confidence": 0-100,
    "severity": "crisis | high | moderate | low",
    "assistance_needed": ["list", "of", "assistance", "types"]
}}"""
    
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    params = {"key": api_key}
    
    try:
        response = requests.post(endpoint, headers=headers, params=params, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            classified = json.loads(match.group(0))
            
            primary = classified.get('primary_category', 'STRESS').upper()
            if primary not in PRIMARY_CATEGORIES:
                primary = 'STRESS'
            
            return {
                'primary_category': primary,
                'specific_category': classified.get('specific_concern', 'general'),
                'confidence': float(classified.get('confidence', 70)) / 100.0,
                'all_matches': [],
                'assistance_types': classified.get('assistance_needed', ['Counseling']),
                'severity': classified.get('severity', 'moderate'),
                'method': 'gemini',
                'matched_keywords': []
            }
    except Exception as e:
        print(f"Gemini classification error: {e}")
    
    return _fallback_classification()


def _fallback_classification():
    """Fallback classification when APIs unavailable."""
    return {
        'primary_category': 'STRESS',
        'specific_category': 'general_stress',
        'confidence': 0.50,
        'all_matches': [],
        'assistance_types': ['Counseling'],
        'severity': 'moderate',
        'method': 'fallback',
        'matched_keywords': []
    }


def get_category_specific_questions(category_result):
    """Get follow-up questions specific to the detected category."""
    primary = category_result.get('primary_category', '')
    
    questions = {
        'CRISIS': [
            "Are you safe right now?",
            "Do you have a plan to hurt yourself?",
            "Is there someone who can be with you right now?"
        ],
        'TRAUMA': [
            "How long ago did this happen?",
            "Are you currently in a safe situation?",
            "Have you experienced flashbacks or nightmares?"
        ],
        'MOOD': [
            "How long have you been feeling this way?",
            "Has this affected your daily activities (sleep, eating, work)?",
            "Have you felt this way before?"
        ],
        'ANXIETY': [
            "How often do you experience these feelings?",
            "Are there specific situations that trigger your anxiety?",
            "Does this affect your daily life (school, relationships, sleep)?"
        ],
        'SUBSTANCE': [
            "What substance(s) are you concerned about?",
            "How often are you using?",
            "Are you looking for detox, outpatient, or ongoing support?"
        ],
        'EATING': [
            "How is this affecting your eating patterns?",
            "Have others expressed concern about your eating?",
            "How long has this been going on?"
        ],
        'ADJUSTMENT': [
            "What change are you adjusting to?",
            "How long have you been experiencing this?",
            "What's been the hardest part?"
        ],
        'STRESS': [
            "What's the main source of your stress right now?",
            "How is this affecting your daily life?",
            "What coping strategies have you tried?"
        ],
        'GRIEF': [
            "Who or what did you lose?",
            "How recent was this loss?",
            "Do you have support from family or friends?"
        ],
        'IDENTITY': [
            "Can you tell me more about what you're experiencing?",
            "Do you have a supportive community or safe people to talk to?",
            "How is this affecting your daily life?"
        ],
        'RELATIONSHIP': [
            "Who is this conflict with (family, partner, roommate, friend)?",
            "How long has this been going on?",
            "Is there any safety concern in this situation?"
        ],
        'ACADEMIC': [
            "What specific academic challenge are you facing?",
            "How is this affecting your wellbeing?",
            "Have you spoken with an academic advisor?"
        ],
        'RESOURCE': [
            "What resources are you lacking access to?",
            "Is this affecting your ability to stay in school?",
            "Have you connected with any campus support services?"
        ]
    }
    
    return questions.get(primary, [
        "Can you tell me more about what you're experiencing?",
        "How long has this been going on?",
        "How is this affecting your daily life?"
    ])


def get_category_empathy_response(category_result, user_name):
    """Get an empathetic response tailored to the specific category."""
    primary = category_result.get('primary_category', '')
    
    responses = {
        'CRISIS': f"{user_name}, I'm really glad you reached out. What you're feeling is serious, and you don't have to face this alone. Your safety matters most right now.",
        'TRAUMA': f"{user_name}, thank you for trusting me with this. What you've experienced is significant, and it takes courage to talk about it. You deserve support.",
        'MOOD': f"{user_name}, I hear you. Depression can make everything feel heavy and overwhelming. Reaching out like this takes real strength, and I want you to know there is help available.",
        'ANXIETY': f"{user_name}, anxiety can be really exhausting to deal with. Thank you for sharing what you're going through. Let's find you some support that can help you manage these feelings.",
        'SUBSTANCE': f"{user_name}, I appreciate you opening up about this. Recognizing you need help is an important first step, and there are people who specialize in supporting you through this.",
        'EATING': f"{user_name}, thank you for sharing something so personal. Your relationship with food and your body can be complicated, and you deserve compassionate support.",
        'ADJUSTMENT': f"{user_name}, transitions and changes can be really hard to navigate. It's completely normal to struggle during these times, and support can make a real difference.",
        'STRESS': f"{user_name}, it sounds like you're carrying a lot right now. Stress can build up and affect us in so many ways. Let's find you some support to help lighten the load.",
        'GRIEF': f"{user_name}, I'm so sorry for what you're going through. Grief is one of the hardest things we experience, and you don't have to navigate it alone.",
        'IDENTITY': f"{user_name}, exploring and understanding who you are can be both challenging and important. Thank you for sharing this with me. You deserve to feel supported and accepted.",
        'RELATIONSHIP': f"{user_name}, relationships can be a source of both support and stress. What you're dealing with sounds difficult, and it's okay to need help navigating it.",
        'ACADEMIC': f"{user_name}, academic challenges can really weigh on us, especially when we put pressure on ourselves. Let's find you support both for the academic side and for how you're feeling.",
        'RESOURCE': f"{user_name}, dealing with practical barriers on top of everything else can feel overwhelming. There are resources available to help, and you deserve access to them."
    }
    
    return responses.get(primary, f"{user_name}, thank you for sharing what's on your mind. I'm here to help you find the support you need.")


def format_classification_summary(category_result):
    """Format the classification result for display."""
    primary = category_result.get('primary_category', 'Unknown')
    specific = category_result.get('specific_category', 'general')
    confidence = category_result.get('confidence', 0)
    severity = category_result.get('severity', 'moderate')
    assistance = category_result.get('assistance_types', ['Counseling'])
    
    cat_info = PRIMARY_CATEGORIES.get(primary, {})
    cat_name = cat_info.get('name', primary)
    cat_color = cat_info.get('color', '⚪')
    
    severity_icons = {
        'crisis': '🔴 CRISIS',
        'high': '🟠 High Priority',
        'moderate': '🟡 Moderate',
        'low': '🟢 Standard'
    }
    
    summary = f"""
┌{'─'*66}┐
│{' ASSESSMENT SUMMARY '.center(66)}│
├{'─'*66}┤
│ {cat_color} Category: {cat_name.ljust(52)}│
│    Specific: {specific.replace('_', ' ').title().ljust(50)}│
│    Severity: {severity_icons.get(severity, severity).ljust(50)}│
│    Confidence: {f'{confidence:.0%}'.ljust(48)}│
├{'─'*66}┤
│ Recommended Support:{' '*45}│"""
    
    for assist in assistance[:3]:
        summary += f"\n│    • {assist.ljust(58)}│"
    
    summary += f"\n└{'─'*66}┘"
    
    return summary


# =====================================================
# LANGUAGE DETECTION
# =====================================================

def detect_language(text):
    """Detect if the user is speaking Spanish, Chinese, or English."""
    text_lower = text.lower()
    
    # Check for Chinese characters
    chinese_char_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    if chinese_char_count >= 2:
        return 'zh'
    
    # Spanish indicators
    SPANISH_INDICATORS = [
        'estoy', 'siento', 'tengo', 'necesito', 'quiero', 'puedo',
        'muy', 'porque', 'cuando', 'como', 'donde', 'quien',
        'ansiosa', 'ansioso', 'triste', 'deprimido', 'deprimida',
        'ayuda', 'salud mental', 'me siento', 'no puedo'
    ]
    
    spanish_matches = sum(1 for indicator in SPANISH_INDICATORS if indicator in text_lower)
    if spanish_matches >= 2:
        return 'es'
    
    strong_indicators = ['estoy', 'siento', 'tengo', 'necesito', 'me siento', 'no puedo']
    if any(indicator in text_lower for indicator in strong_indicators):
        return 'es'
    
    return 'en'


# =====================================================
# RESOURCE DISPLAY FUNCTIONS
# =====================================================

def display_emergency_resources():
    """Display comprehensive emergency mental health resources."""
    return """
╔════════════════════════════════════════════════════════════════════╗
║                  🆘 EMERGENCY MENTAL HEALTH RESOURCES              ║
╚════════════════════════════════════════════════════════════════════╝

If you are in immediate danger or having thoughts of self-harm:

📞 **National Suicide Prevention Lifeline**
   Call or Text: 988
   Available: 24/7 - Free and Confidential

💬 **Crisis Text Line**
   Text: HOME to 741741
   Available: 24/7 - Free Crisis Counseling

🚨 **Emergency Services**
   Call: 911 for immediate emergency assistance

💙 **The Adam Project** (Free Mental Health Provider Directory)
   Website: www.TheAdamProject.org
   1,300+ free mental health providers across America

🏥 **SAMHSA National Helpline**
   Call: 1-800-662-HELP (4357)
   Available: 24/7 - Free, Confidential

═══════════════════════════════════════════════════════════════════════
Remember: You are not alone. Help is available right now.
═══════════════════════════════════════════════════════════════════════
"""


def translate_crisis_resources_to_spanish():
    """Return crisis resources translated to Spanish."""
    return """
╔══════════════════════════════════════════════════════════════════════╗
║           🆘 APOYO INMEDIATO DISPONIBLE 24/7 (Recursos en EE.UU.)    ║
╚══════════════════════════════════════════════════════════════════════╝

📞 **988 - Línea de Prevención del Suicidio y Crisis**
    Llama o envía un mensaje de texto al 988
    Servicio en español disponible

💬 **Línea de Texto en Crisis**
    Envía HOLA al 741741

🚨 **Servicios de Emergencia**
    Llama al 911

📞 **Línea Nacional en Español**
    1-888-628-9454 (24/7)
══════════════════════════════════════════════════════════════════════
"""


def translate_crisis_resources_to_chinese():
    """Return crisis resources translated to Chinese."""
    return """
╔══════════════════════════════════════════════════════════════════════╗
║           🆘 24/7紧急心理健康支持 (美国资源)                          ║
╚══════════════════════════════════════════════════════════════════════╝

📞 **988 - 全国自杀预防生命线**
    拨打或发短信至 988

💬 **危机短信热线**
    发送 HOME 到 741741

🚨 **紧急服务**
    拨打 911

📞 **中文服务热线**
    1-800-273-8255 (24/7)
══════════════════════════════════════════════════════════════════════
"""


# =====================================================
# STATE MAPPING AND LOCATION PARSING
# =====================================================

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
    """Parse a location string that may contain city and/or state."""
    if not location_string:
        return None, None
    
    input_lower = location_string.lower().strip()
    tokens = input_lower.replace(',', ' ').split()
    found_state = None
    remaining_text = input_lower
    
    if tokens:
        last_token = tokens[-1].upper()
        if len(last_token) == 2 and last_token.isalpha():
            if last_token in STATE_MAPPING.values():
                found_state = last_token
                remaining_text = ' '.join(tokens[:-1]).strip()
    
    if not found_state:
        sorted_states = sorted(STATE_MAPPING.items(), key=lambda x: len(x[0]), reverse=True)
        for full_name, abbrev in sorted_states:
            if full_name in ['new york', 'washington']:
                if input_lower.endswith(full_name) or input_lower.endswith(f', {full_name}'):
                    found_state = abbrev
                    remaining_text = input_lower.replace(full_name, '').strip()
                    break
            else:
                if full_name in input_lower:
                    found_state = abbrev
                    remaining_text = input_lower.replace(full_name, '').strip()
                    break
    
    city = remaining_text.replace(',', '').strip()
    if city:
        if city.isupper() and len(city) <= 3:
            city = city.upper()
        else:
            city = ' '.join(word.capitalize() for word in city.split())
    else:
        city = None
    
    return city, found_state


# =====================================================
# FACILITY SEARCH FUNCTIONS
# =====================================================

def fast_search_scored_csv(scored_csv_path, city=None, state=None, zipcode=None, top_n=5):
    """Lightweight search over a pre-scored CSV file."""
    usecols = [
        'name', 'street', 'city', 'state', 'zipcode', 'zip', 'phone',
        'overall_care_needs_score', 'affordability_score', 'crisis_care_score'
    ]
    
    try:
        df = pd.read_csv(
            scored_csv_path,
            dtype={'name': str, 'street': str, 'city': str, 'state': str,
                   'zipcode': str, 'zip': str, 'phone': str},
            usecols=lambda x: x in usecols,
            na_values=['', 'NA', 'N/A'],
            low_memory=True
        )
    except Exception as e:
        print(f"Warning: Optimized loading failed: {e}")
        df = pd.read_csv(scored_csv_path, dtype=str)
        for col in ['overall_care_needs_score', 'affordability_score', 'crisis_care_score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    try:
        if state:
            state_code = state.upper()
            state_mask = df['state'].str.strip().str.upper() == state_code
            df = df[state_mask]
        
        if city:
            city_normalized = city.lower().strip()
            city_mask = df['city'].str.lower().str.strip() == city_normalized
            df = df[city_mask]
        
        if zipcode:
            target_zip = ''.join(ch for ch in str(zipcode) if ch.isdigit())
            if target_zip:
                df_before_zip = df.copy()
                zip_mask = pd.Series(False, index=df.index)
                for zcol in ('zipcode', 'zip'):
                    if zcol in df.columns:
                        normalized = df[zcol].fillna('').astype(str).str.replace(r'\D', '', regex=True)
                        zip_mask = zip_mask | normalized.str.startswith(target_zip)
                df = df[zip_mask]
                if df.empty:
                    df = df_before_zip
        
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
            df = df.head(top_n)
        
        records = df.replace({np.nan: None}).to_dict(orient='records')
    except Exception as e:
        print(f"Warning: Error during filtering: {e}")
        records = []
    
    return records


def format_facility_results(facilities, output_format='simple'):
    """Format facility results for end users."""
    if not facilities:
        return "No facilities found."
    
    normalized = []
    for f in facilities:
        nf = dict(f) if isinstance(f, dict) else f
        for zkey in ('zip', 'zipcode'):
            if zkey in nf and nf[zkey] is not None:
                try:
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
        
        if 'phone' in nf and nf['phone'] is not None:
            nf['phone'] = str(nf['phone'])
        
        normalized.append(nf)
    
    if output_format == 'json':
        return json.dumps(normalized, indent=2, default=str)
    
    lines = []
    for i, f in enumerate(normalized, 1):
        name = f.get('name') or f.get('facility_name') or 'Unknown Facility'
        street = f.get('address') or f.get('street') or ''
        city = f.get('city') or ''
        state = f.get('state') or ''
        zipcode = f.get('zip') or f.get('zipcode') or ''
        phone = f.get('phone') or 'Phone not available'
        
        score = f.get('overall_care_needs_score') or f.get('score')
        try:
            score_str = f"{float(score):.1f}/10" if score is not None else 'N/A'
        except Exception:
            score_str = str(score)
        
        lines.append(f"{i}. {name}")
        if street:
            lines.append(f"   📍 {street}")
        loc_line = f"   📍 {city}, {state}"
        if zipcode:
            loc_line += f" {zipcode}"
        lines.append(loc_line)
        lines.append(f"   📞 {phone}")
        lines.append(f"   ⭐ Score: {score_str}")
        lines.append("")
    
    return "\n".join(lines).strip()


# =====================================================
# HARBOR CHATBOT FUNCTIONS
# =====================================================

def harbor_greet():
    """Harbor introduces itself and asks for the user's name."""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
        api_key = config.get("GEMINI_API_KEY")
    except Exception:
        return "Hello! I'm Harbor, and I'm here to help you find the support you need. What's your name?"
    
    if not api_key:
        return "Hello! I'm Harbor, and I'm here to help you find the support you need. What's your name?"
    
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    system_prompt = """You are Harbor, a warm and empathetic mental health assistant. Start by greeting the user warmly and asking for their name. Keep your greeting brief and friendly (2-3 sentences max)."""
    
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"role": "user", "parts": [{"text": system_prompt}]}]}
    params = {"key": api_key}
    
    try:
        response = requests.post(endpoint, headers=headers, params=params, json=payload)
        response.raise_for_status()
        result = response.json()
        greeting = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        return greeting
    except Exception:
        return "Hello! I'm Harbor, and I'm here to help you find the support you need. What's your name?"


def harbor_ask_concern(user_name, conversation_history):
    """After getting the user's name, Harbor asks what's on their mind."""
    return f"Hi {user_name}, it's nice to meet you. What's on your mind today? How can I help you?"


def harbor_respond_with_empathy(user_name, user_concern, conversation_history, language='en'):
    """
    Provides empathetic acknowledgment with detailed category classification.
    """
    # Get detailed classification
    category_result = classify_mental_health_category(user_concern, conversation_history)
    
    primary = category_result['primary_category']
    severity = category_result['severity']
    confidence = category_result['confidence']
    
    # Check for crisis first
    if primary == 'CRISIS' or severity == 'crisis':
        crisis_result = detect_crisis_hybrid(user_concern)
        if crisis_result['is_crisis']:
            severity_assessment = assess_crisis_severity(user_concern, crisis_result)
            
            # Display emergency resources
            if language == 'es':
                print("\n" + translate_crisis_resources_to_spanish())
            elif language == 'zh':
                print("\n" + translate_crisis_resources_to_chinese())
            else:
                print("\n" + display_emergency_resources())
            
            print(f"\n🚢 Harbor: {user_name}, I'm really glad you reached out to me.")
            print("          What you're feeling is serious, and I want you to know")
            print("          you're not alone. Please use the resources above for")
            print("          immediate support.\n")
            
            return {
                'is_crisis': True,
                'category_result': category_result,
                'severity': severity_assessment['severity'],
                'urgency_score': severity_assessment['urgency_score']
            }
    
    # Non-crisis: Display classification and empathetic response
    print(format_classification_summary(category_result))
    
    empathy_message = get_category_empathy_response(category_result, user_name)
    print(f"\n🚢 Harbor: {empathy_message}\n")
    
    if category_result.get('matched_keywords'):
        keywords = category_result['matched_keywords'][:3]
        print(f"   (I noticed you mentioned: {', '.join(keywords)})\n")
    
    return {
        'is_crisis': False,
        'category_result': category_result,
        'severity': severity,
        'confidence': confidence
    }


def collect_category_specific_info(category_result, user_name):
    """Ask follow-up questions based on the detected category."""
    questions = get_category_specific_questions(category_result)
    primary = category_result['primary_category']
    
    print(f"\n🚢 Harbor: {user_name}, to find the best support for you,")
    print("          I'd like to understand a bit more.\n")
    
    responses = {}
    
    for i, question in enumerate(questions[:2]):
        print(f"🚢 Harbor: {question}")
        response = input("You: ").strip()
        if response:
            responses[f'question_{i+1}'] = {'question': question, 'answer': response}
        print()
    
    symptom_details = []
    for key, data in responses.items():
        symptom_details.append(f"{data['question']} → {data['answer']}")
    
    return {
        'primary_category': primary,
        'specific_category': category_result['specific_category'],
        'severity': category_result['severity'],
        'follow_up_responses': responses,
        'symptom_details': "; ".join(symptom_details),
        'assistance_types': category_result['assistance_types']
    }


# =====================================================
# MAIN PIPELINE
# =====================================================

def run_pipeline():
    """Main pipeline orchestration with Harbor chatbot."""
    
    print("\n" + "═"*70)
    print("  🚢 HARBOR - Mental Health Support Assistant".center(70))
    print("  Enhanced with 80-Scenario Category Taxonomy".center(70))
    print("═"*70)
    print("\nWelcome! I'm here to listen and help you find the support you need.")
    print()
    
    conversation_history = []
    
    # Step 1: Greeting
    harbor_greeting = harbor_greet()
    print(f"🚢 Harbor: {harbor_greeting}\n")
    conversation_history.append({'role': 'BOT', 'message': harbor_greeting})
    
    user_name_input = ""
    while not user_name_input:
        user_name_input = input("You: ").strip()
        if not user_name_input:
            print("🚢 Harbor: I'd love to know your name so I can help you better.\n")
    
    conversation_history.append({'role': 'USER', 'message': user_name_input})
    
    # Extract name
    user_name = user_name_input
    name_match = re.search(r'(?:name is |i\'m |im |call me )([a-zA-Z]+)', user_name_input.lower())
    if name_match:
        user_name = name_match.group(1).capitalize()
    elif ' ' not in user_name_input and len(user_name_input) < 20:
        user_name = user_name_input.capitalize()
    
    # Step 2: Ask about concern
    print()
    concern_prompt = harbor_ask_concern(user_name, conversation_history)
    print(f"🚢 Harbor: {concern_prompt}\n")
    conversation_history.append({'role': 'BOT', 'message': concern_prompt})
    
    user_concern = ""
    while not user_concern:
        user_concern = input("You: ").strip()
        if not user_concern:
            print("🚢 Harbor: Please share what's on your mind - I'm here to listen and help.\n")
    
    conversation_history.append({'role': 'USER', 'message': user_concern})
    
    # Step 3: Detect language
    user_language = detect_language(user_concern)
    
    # Step 4: Classify and respond with empathy
    print("\n" + "─"*70)
    print("⚙️  Analyzing your concerns...")
    print("─"*70 + "\n")
    
    empathy_response = harbor_respond_with_empathy(
        user_name, user_concern, conversation_history, user_language
    )
    
    is_crisis = empathy_response.get('is_crisis', False)
    category_result = empathy_response.get('category_result', {})
    
    # Step 5: Collect category-specific info (if not crisis)
    if not is_crisis:
        category_info = collect_category_specific_info(category_result, user_name)
    else:
        category_info = category_result
    
    # Step 6: Collect location
    print("\n" + "┌" + "─"*68 + "┐")
    print("│" + " 📍 Location Information ".center(68) + "│")
    print("└" + "─"*68 + "┘\n")
    
    location_prompt = "🚢 Harbor: To find support near you, what city and state are you in?\n          (e.g., Charlotte, NC)\n\nYou: "
    location_input = input(location_prompt).strip()
    
    city, state = parse_location_input(location_input)
    
    if not state:
        state_input = input("🚢 Harbor: What state? ").strip()
        _, state = parse_location_input(f"City {state_input}")
        if not state:
            state = state_input.upper() if len(state_input) == 2 else state_input
    
    if not city:
        city = input("🚢 Harbor: What city? ").strip().title()
    
    # Step 7: Collect insurance
    print()
    insurance_input = input("🚢 Harbor: Do you have health insurance? (yes/no) ").strip().lower()
    has_insurance = insurance_input.startswith('y')
    
    insurance_type = ''
    if has_insurance:
        insurance_type = input("🚢 Harbor: What's your insurance provider? ").strip()
    
    # Step 8: Build classification
    classification = {
        'category': PRIMARY_CATEGORIES.get(category_info.get('primary_category', 'STRESS'), {}).get('name', 'Mental health'),
        'primary_category': category_info.get('primary_category', 'STRESS'),
        'specific_category': category_info.get('specific_category', 'general'),
        'confidence': category_result.get('confidence', 0.7),
        'severity': category_info.get('severity', 'moderate'),
        'symptoms': f"{user_concern}. {category_info.get('symptom_details', '')}",
        'assistance_types': category_info.get('assistance_types', ['Counseling']),
        'location': {'city': city, 'state': state},
        'insurance': {'has_insurance': has_insurance, 'provider': insurance_type}
    }
    
    # Step 9: Display summary
    print("\n" + "┌" + "─"*68 + "┐")
    print("│" + " 📋 Summary ".center(68) + "│")
    print("└" + "─"*68 + "┘")
    
    cat_info = PRIMARY_CATEGORIES.get(classification['primary_category'], {})
    print(f"\n{cat_info.get('color', '⚪')} Category: {classification['category']}")
    print(f"   Specific: {classification['specific_category'].replace('_', ' ').title()}")
    print(f"   Severity: {classification['severity'].title()}")
    print(f"📍 Location: {city}, {state}")
    print(f"💳 Insurance: {'Yes - ' + insurance_type if has_insurance else 'No'}")
    
    # Step 10: Search facilities
    print("\n" + "─"*70)
    print("🔍 Searching for matching facilities...")
    print("─"*70 + "\n")
    
    scored_csv = root_dir / "datasets" / "all_facilities_scored.csv"
    
    if scored_csv.exists():
        facilities = fast_search_scored_csv(str(scored_csv), city=city, state=state, top_n=5)
        
        if facilities:
            print(f"✓ Found {len(facilities)} facilities\n")
            print(format_facility_results(facilities))
        else:
            print("⚠️  No facilities found matching your criteria.")
            print("   Try broadening your search or contact 988 for immediate support.")
    else:
        print("⚠️  Facility database not found.")
        print("   Please ensure all_facilities_scored.csv exists in /datasets/")
    
    # Step 11: Closing message
    print("\n" + "═"*70)
    print("  ✅ SEARCH COMPLETE".center(70))
    print("═"*70)
    
    empathy_closing = get_category_empathy_response(category_result, user_name)
    print(f"\n🚢 Harbor: {user_name}, thank you for trusting me with this.")
    print("          Remember, seeking help is a sign of strength.")
    print("\n          💙 You deserve support, and it's out there.")
    print("          📞 If you need to talk to someone right away: call 988")
    print("─"*70 + "\n")
    
    return {
        'status': 'success',
        'classification': classification,
        'facilities': facilities if 'facilities' in dir() else None
    }


def main():
    """Main entry point."""
    try:
        result = run_pipeline()
        print(f"\n[Pipeline completed: {result['status']}]")
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted. Goodbye!")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()