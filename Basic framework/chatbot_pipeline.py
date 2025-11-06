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
                city_mask = df['city'].str.contains(city, case=False, na=False)
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
# Mock Classifier (Temporary - Replace with LLM Later)
# =====================================================

def mock_classify_conversation(conversation_history):
    """
    LLM-backed conversation handler.

    Behavior:
    - If OpenAI python package is available and OPENAI_API_KEY is set, use the Chat API
      to analyze the provided `conversation_history`, conduct up to a small number
      of follow-up questions (interactive via input()) to extract missing fields,
      and return a structured dictionary.
    - If OpenAI isn't available or API key is missing, fall back to a lightweight
      heuristic that summarizes user messages (keeps prior mock behavior).

    Returns:
        dict: must contain 'category', 'confidence', 'user_input'. May also include
              optional fields: 'symptoms', 'location' (dict), 'insurance' (dict).
    """

    # Gather existing user messages into a single text blob
    user_messages = [msg.get('message', '') for msg in conversation_history if msg.get('role') == 'USER']
    conversation_text = "\n".join(user_messages).strip()

    # Try to use OpenAI ChatCompletion if available
    try:
        import openai
        OPENAI_KEY = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_APIKEY')
        if not OPENAI_KEY:
            raise RuntimeError('OPENAI_API_KEY not found in environment')

        openai.api_key = OPENAI_KEY

        system_prompt = (
            "You are a helpful clinical intake assistant. Conduct a natural, empathetic "
            "conversation only to the extent needed to extract the following structured "
            "information from the user: category (one short category label), confidence "
            "(0-1 or 0-100), user_input (short summary), symptoms (brief), location {city, state}, "
            "insurance {has_insurance: bool, provider: str or empty}. If information is missing, "
            "ask one concise follow-up question. After you have the information, reply with ONLY a JSON object"
            " (no additional text) containing these fields. Make confidence numeric.")

        # Build chat history for the model: include prior conversation as user/system turns
        messages = [
            {'role': 'system', 'content': system_prompt},
        ]

        # Append conversation history preserving roles
        for msg in conversation_history:
            role = msg.get('role', 'USER')
            content = msg.get('message', '')
            # Map local roles to chat roles
            chat_role = 'user' if role.upper() in ('USER', 'U') else 'assistant' if role.upper() in ('BOT', 'SYSTEM') else 'user'
            messages.append({'role': chat_role, 'content': content})

        # We'll allow a small interactive loop: up to 3 follow-ups to collect required fields
        max_rounds = 3
        for round_i in range(max_rounds):
            # Choose model: prefer gpt-4 if available, else gpt-3.5-turbo
            model = os.getenv('OPENAI_MODEL', 'gpt-4')
            try:
                resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.2)
            except Exception:
                # fallback to cheaper model
                resp = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages, temperature=0.2)

            assistant_msg = resp['choices'][0]['message']['content'].strip()

            # Try to extract JSON from assistant message
            parsed = None
            try:
                parsed = json.loads(assistant_msg)
            except Exception:
                # try to find a JSON substring
                import re
                m = re.search(r"\{(?:.|\n)*\}", assistant_msg)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = None

            # If parsed and contains required keys, return it
            if isinstance(parsed, dict) and all(k in parsed for k in ('category', 'confidence', 'user_input')):
                # Normalize location/insurance if present
                return parsed

            # If the assistant returned a follow-up question as text, present to user
            # Heuristic: treat non-JSON as a follow-up question
            follow_up = None
            if not parsed:
                follow_up = assistant_msg
            else:
                # parsed but missing fields -> ask assistant to generate single follow-up
                follow_up = (parsed.get('follow_up_question') or
                             parsed.get('clarifying_question') or
                             "Could you tell me your city and state? (or type 'skip')")

            # Ask user and append reply to messages
            try:
                reply = input(f"{follow_up}\n> ").strip()
            except Exception:
                # non-interactive environment: break and fallback
                break

            messages.append({'role': 'user', 'content': reply})

        # If loop ends without structured full result, try one last analysis pass (no follow-ups)
        final_resp = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages + [{'role': 'system', 'content': 'Now produce the requested JSON with the fields.'}],
            temperature=0.0
        )
        final_text = final_resp['choices'][0]['message']['content'].strip()
        try:
            final_parsed = json.loads(final_text)
            if isinstance(final_parsed, dict) and 'category' in final_parsed:
                return final_parsed
        except Exception:
            pass

        # If still no structured result, fall through to fallback below

    except Exception as e:
        # Any failure to use the OpenAI path should not crash the pipeline.
        print(f"LLM integration unavailable or failed: {e}")

    # Fallback heuristic: summarize user messages and return low-confidence classification
    combined_input = conversation_text or ""
    return {
        'category': 'Mental health',
        'confidence': 0.6,
        'user_input': combined_input[:100],
        'symptoms': combined_input,
        'location': {},
        'insurance': {}
    }


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
    
    # Collect and normalize city. We apply a small heuristic:
    # - If the user typed an all-uppercase short token (e.g., 'DC' or 'NYC'), keep it.
    # - Otherwise use title-casing for consistency (e.g., 'new york' -> 'New York').
    city_raw = input("What city are you in? ").strip()
    if city_raw and city_raw.isupper() and len(city_raw) <= 3:
        city = city_raw
    else:
        # Basic heuristic: title-case words but preserve existing all-caps words
        parts = []
        for w in city_raw.split():
            parts.append(w if w.isupper() else w.capitalize())
        city = " ".join(parts).strip()

    # Accept either 2-letter codes or full state names (case-insensitive).
    # Normalize to 2-letter uppercase codes when mapping is available.
    state_raw = input("What state are you in? (2-letter code or full name, e.g., NC or North Carolina) ").strip()

    # Allow one re-prompt if the state isn't recognized (better UX).
    attempts = 0
    state = ''
    while attempts < 2:
        if not state_raw:
            # empty input -> keep empty and break
            break

        s = state_raw.strip()
        if len(s) == 2 and s.isalpha():
            state = s.upper()
            break

        mapped = STATE_MAPPING.get(s.lower())
        if mapped:
            state = mapped
            break

        # Not recognized: if first attempt, prompt again; otherwise accept uppercase fallback
        attempts += 1
        if attempts < 2:
            state_raw = input("I couldn't recognize that state. Please enter 2-letter code or full state name (or press Enter to skip): ").strip()
        else:
            # fallback: store uppercase of raw input
            state = s.upper()

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
    
    print("\n" + "="*70)
    print("SEARCHING FOR FACILITIES...")
    print("="*70)
    
    # Use pre-scored CSV if available (fast path)
    print(f"Category: {classification['category']}")
    print(f"Confidence: {classification['confidence']:.0%}")

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

    location_line = f"Location: {city or 'N/A'}, {state or 'N/A'}"
    if zip_code:
        location_line += f" {zip_code}"
    print(location_line)
    print(f"Insurance: {'Yes' if additional_info.get('insurance', {}).get('has_insurance') else 'No'}")

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

            # Determine desired output format for presentation (default 'simple')
            output_format = 'simple'
            if isinstance(additional_info, dict):
                output_format = additional_info.get('output_format', output_format)

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
        import facility_scorer
        # Note: the integration path to run the full scorer can be implemented here
        # (e.g., call facility_scorer.score_csv_file or similar). For now, we simply
        # inform the user and return None.
    except Exception:
        # If importing the scorer fails, keep behavior minimal and inform the user.
        pass

    return None


def run_pipeline():
    """
    Main pipeline orchestration
    
    This is the master function that runs the complete flow:
    1. Collect conversation via chatbot interface
    2. Classify the conversation (mock or real LLM)
    3. Route using group2_router
    4. If Group 3 → collect info → match facilities
    5. If not Group 3 → hand off to appropriate group
    """
    
    print("\n" + "="*70)
    print("  MENTAL HEALTH CHATBOT PIPELINE".center(70))
    print("  AAI6600 Fall 2025 - Group 3".center(70))
    print("="*70)
    print("\nWelcome! This system will help connect you with mental health resources.")
    print("Let's start with a brief conversation to understand your needs.\n")
    
    # Step 1: Collect conversation
    print("[Step 1] Starting conversation...")
    
    # Mock conversation for now (will be replaced with chatbot_interface)
    mock_conversation = [
        {'role': 'BOT', 'message': 'How can I help you today?'},
        {'role': 'USER', 'message': "I've been feeling really anxious and need help"},
        {'role': 'BOT', 'message': 'Can you tell me more about what you\'re experiencing?'},
        {'role': 'USER', 'message': 'I have panic attacks and trouble sleeping'}
    ]
    
    print("✓ Conversation collected\n")
    
    # Step 2: Classify the conversation
    print("[Step 2] Classifying mental health needs...")
    from data_adapter import adapt_mock_output

    raw_classification = mock_classify_conversation(mock_conversation)
    classification = adapt_mock_output(raw_classification)
    print(f"✓ Classification: {classification['category']} ({classification['confidence']:.0%} confidence)\n")
    
    # Step 3: Route using group2_router
    print("[Step 3] Routing to appropriate group...")
    is_ours, routing_decision = handle_group2_input(classification)
    print(f"✓ {routing_decision['message']}\n")
    
    # Step 4: Handle based on routing decision
    if not is_ours:
        # Not our category - hand off
        print(f"→ This request should be handled by {routing_decision['branch']}")
        print("→ Passing to appropriate team...\n")
        return {
            'status': 'handed_off',
            'branch': routing_decision['branch'],
            'classification': classification
        }
    
    # Step 5: Group 3 handles - collect additional info
    print("[Step 4] Collecting additional information for facility matching...")
    additional_info = collect_additional_info()
    print("✓ Information collected\n")

    # Ask user preferred output view (simple vs json). Default to simple.
    try:
        view_choice = input("Choose output view - (1) Simple (non-technical)  (2) JSON (technical) [default 1]: ").strip()
    except EOFError:
        # non-interactive runs: default to simple
        view_choice = ''

    if view_choice == '2' or view_choice.lower() in ('json', 'j'):
        additional_info.setdefault('output_format', 'json')
    else:
        additional_info.setdefault('output_format', 'simple')
    
    # Step 6: Match facilities
    print("[Step 5] Finding facility recommendations...")
    facilities = call_facility_matcher(classification, additional_info)
    
    # Step 7: Display results
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    if facilities:
        print(f"\nFound {len(facilities)} recommended facilities:")
        for i, facility in enumerate(facilities, 1):
            print(f"{i}. {facility}")
    else:
        print("\nFacility matching will be implemented next.")
    
    print("\n" + "="*70)
    
    return {
        'status': 'success',
        'classification': classification,
        'additional_info': additional_info,
        'facilities': facilities
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
