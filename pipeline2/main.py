#!/usr/bin/env python3
"""
Pipeline 2 - Main Entry Point (CLI)

Command-line interface for mental health classification
and facility search.

Usage:
    python main.py classify "I'm feeling anxious"
    python main.py search --city Boston --state MA
    python main.py score --input data.csv --output scored.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

from core.mental_health_classifier import MentalHealthClassifier
from core.facility_scorer import FacilityScorer
from core.anti_hallucination import AntiHallucinationValidator


def cmd_classify(args):
    """Classify user input"""
    print("=" * 70)
    print("MENTAL HEALTH INTENT CLASSIFICATION")
    print("=" * 70)
    
    # Load classifier
    print("\nLoading classifier...")
    classifier = MentalHealthClassifier()
    
    # Classify
    print(f"\nInput: {args.text}")
    result = classifier.classify_with_response(args.text)
    
    print(f"\n{'=' * 70}")
    print("RESULT")
    print("=" * 70)
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Crisis: {result['is_crisis']}")
    
    if result['is_crisis']:
        print(f"\n🚨 CRISIS DETECTED")
        print(result['response'])
    else:
        print(f"\nResponse: {result['response']}")


def cmd_search(args):
    """Search facilities"""
    print("=" * 70)
    print("FACILITY SEARCH")
    print("=" * 70)
    
    # Load data
    data_dir = Path(__file__).parent / "data"
    facilities_file = data_dir / "all_facilities_scored.csv"
    
    if not facilities_file.exists():
        print(f"\n✗ Error: No facility data found at {facilities_file}")
        print("   Please add facility data to data/ folder")
        return
    
    print(f"\nLoading facilities from {facilities_file.name}...")
    df = pd.read_csv(facilities_file, low_memory=False)
    print(f"✓ Loaded {len(df):,} facilities")
    
    # Load scorer
    print("\nLoading scorer...")
    scorer = FacilityScorer()
    
    # Search
    print(f"\nSearching for:")
    if args.city:
        print(f"  City: {args.city}")
    if args.state:
        print(f"  State: {args.state}")
    if args.needs:
        print(f"  Needs: {', '.join(args.needs)}")
    
    results = scorer.get_top_facilities(
        df,
        n=args.top_n,
        city=args.city,
        state=args.state,
        needs=args.needs
    )
    
    # Display results
    print(f"\n{'=' * 70}")
    print(f"FOUND {len(results)} FACILITIES")
    print("=" * 70)
    
    for idx, (_, facility) in enumerate(results.iterrows(), 1):
        print(f"\n{idx}. {facility.get('name', 'Unknown')}")
        print(f"   {facility.get('street', 'N/A')}")
        print(f"   {facility.get('city', 'N/A')}, {facility.get('state', 'N/A')} {facility.get('zipcode', '')}")
        print(f"   Phone: {facility.get('phone', 'N/A')}")
        print(f"   Score: {facility.get('overall_care_needs_score', 0):.1f}/10")


def cmd_score(args):
    """Score facilities from CSV"""
    print("=" * 70)
    print("FACILITY SCORING")
    print("=" * 70)
    
    # Check input file
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"\n✗ Error: Input file not found: {input_file}")
        return
    
    # Load data
    print(f"\nLoading {input_file.name}...")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} facilities")
    
    # Load scorer
    print("\nInitializing scorer...")
    scorer = FacilityScorer()
    
    # Score
    print("\nScoring facilities...")
    df_scored = scorer.score_facilities(df)
    
    # Save
    output_file = Path(args.output)
    print(f"\nSaving to {output_file}...")
    df_scored.to_csv(output_file, index=False)
    print(f"✓ Saved {len(df_scored):,} scored facilities")
    
    # Display summary
    print(f"\n{'=' * 70}")
    print("SCORING SUMMARY")
    print("=" * 70)
    print(f"Average overall score: {df_scored['overall_care_needs_score'].mean():.2f}/10")
    print(f"High scoring (≥7.5): {(df_scored['overall_care_needs_score'] >= 7.5).sum():,} facilities")
    print("=" * 70)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Pipeline 2 - Mental Health Classification & Facility Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py classify "I'm feeling anxious"
  python main.py search --city Boston --state MA
  python main.py search --city "New York" --state NY --needs crisis affordable
  python main.py score --input facilities.csv --output scored.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify mental health intent')
    classify_parser.add_argument('text', help='User input text to classify')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search mental health facilities')
    search_parser.add_argument('--city', help='City name')
    search_parser.add_argument('--state', help='State code (e.g., MA, CA)')
    search_parser.add_argument('--needs', nargs='+', help='Special needs (e.g., crisis affordable)')
    search_parser.add_argument('--top-n', type=int, default=5, help='Number of results (default: 5)')
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Score facilities from CSV')
    score_parser.add_argument('--input', required=True, help='Input CSV file')
    score_parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    if args.command == 'classify':
        cmd_classify(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'score':
        cmd_score(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)