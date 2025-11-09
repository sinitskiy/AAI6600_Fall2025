#!/usr/bin/env python3
"""Quick test to verify location parsing works with user's inputs"""

from chatbot_pipeline import parse_location_input

# Test cases from user's actual runs
test_cases = [
    'charlotte north carolina',
    'charlotte, north Carolina', 
    'Charlotte, NC',
    'charlotte nc',
    'CHARLOTTE NC',
]

print('Testing location parsing with actual user inputs:')
print('=' * 60)
for test in test_cases:
    city, state = parse_location_input(test)
    status = '✓' if city == 'Charlotte' and state == 'NC' else '✗'
    print(f'{status} Input: "{test}"')
    print(f'  → City: "{city}", State: "{state}"')
    print()
