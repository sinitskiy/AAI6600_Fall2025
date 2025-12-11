# Pipeline 2 - Mental Health Classification & Facility Search

A machine learning pipeline for mental health intent classification and facility recommendation using TF-IDF, SVM, and BERT-based semantic similarity scoring.

## Features

- **Intent Classification**: Classifies user mental health queries using TF-IDF + SVM
- **Crisis Detection**: Automatically detects crisis situations and provides emergency resources
- **Facility Scoring**: Scores mental health facilities using BERT embeddings across 5 dimensions
- **Anti-Hallucination**: Validates results and detects negative evidence to prevent unreliable recommendations
- **Web GUI**: Interactive Streamlit interface for easy use

## Installation

### Prerequisites
- Python 3.8 or higher

### Install Dependencies

```bash
cd pipeline2
pip install -r requirements.txt
```

Or install manually:
```bash
pip install sentence-transformers scikit-learn numpy pandas streamlit plotly matplotlib tqdm
```

## Usage

### Option 1: Command Line Interface (CLI)

#### Classify Mental Health Intent
```bash
python main.py classify "I'm feeling anxious about my exams"
```

#### Search Facilities by Location
```bash
python main.py search --city Boston --state MA
```

#### Search with Special Needs
```bash
python main.py search --city "New York" --state NY --needs crisis affordable --top-n 10
```

#### Score Facilities from CSV
```bash
python main.py score --input facilities.csv --output scored_facilities.csv
```

#### Get Help
```bash
python main.py --help
```

### Option 2: Web GUI (Streamlit)

```bash
python run_gui.py
```

Or run directly:
```bash
streamlit run gui/streamlit_app.py
```

Then open your browser to: **http://localhost:8501**

## Project Structure

```
pipeline2/
├── main.py                 # CLI entry point
├── run_gui.py              # GUI launcher
├── requirements.txt        # Dependencies
├── core/
│   ├── __init__.py
│   ├── mental_health_classifier.py   # Intent classification (TF-IDF + SVM)
│   ├── facility_scorer.py            # Semantic scoring (BERT embeddings)
│   └── anti_hallucination.py         # Validation & negative evidence detection
├── gui/
│   ├── __init__.py
│   └── streamlit_app.py              # Streamlit web interface
└── tests/
    ├── __init__.py
    ├── smoke_test.py                 # Quick functionality test
    ├── test_classifier.py            # Classifier unit tests
    └── test_scorer.py                # Scorer unit tests
```

## Core Components

### 1. Mental Health Classifier
- Uses TF-IDF vectorization + SVM for intent classification
- Trained on mental health conversation patterns
- Detects intents: suicide, sad, anxious, stressed, help, greeting, academic-pressure, loneliness

### 2. Facility Scorer
- Uses sentence-transformers (all-MiniLM-L6-v2) for semantic similarity
- Scores facilities across 5 dimensions:
  - Affordability
  - Crisis care
  - Accessibility
  - Specialization
  - Community integration

### 3. Anti-Hallucination Validator
- Validates scoring reliability based on semantic similarity
- Detects negative evidence (e.g., "cash only", "no insurance accepted")
- Flags low-confidence results with warnings

## CLI Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `classify` | Classify mental health intent | `python main.py classify "I need help"` |
| `search` | Search facilities | `python main.py search --state MA` |
| `score` | Score CSV file | `python main.py score --input data.csv --output scored.csv` |

### Search Options
| Option | Description | Example |
|--------|-------------|---------|
| `--city` | Filter by city name | `--city Boston` |
| `--state` | Filter by state code | `--state MA` |
| `--needs` | Special needs filter | `--needs crisis affordable` |
| `--top-n` | Number of results (default: 5) | `--top-n 10` |

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_classifier.py

# Run smoke test
python tests/smoke_test.py
```

## Example: Python API Usage

```python
from core.mental_health_classifier import MentalHealthClassifier
from core.facility_scorer import FacilityScorer
from core.anti_hallucination import AntiHallucinationValidator
import pandas as pd

# 1. Classify user intent
classifier = MentalHealthClassifier()
result = classifier.classify_with_response("I'm feeling anxious")
print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Crisis: {result['is_crisis']}")

# 2. Score facilities
scorer = FacilityScorer()
df = pd.DataFrame({
    'name': ['Community Mental Health Center', 'City Hospital'],
    'city': ['Boston', 'Boston'],
    'state': ['MA', 'MA'],
    'street': ['123 Main St', '456 Oak Ave']
})
df_scored = scorer.score_facilities(df)
top = scorer.get_top_facilities(df_scored, n=5, state='MA')

# 3. Validate results
validator = AntiHallucinationValidator()
validated = validator.validate_results(top.to_dict('records'))
print(validator.generate_warning_report(validated))
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named sentence_transformers` | `pip install sentence-transformers` |
| `No facility data found` | Add facility CSV to `data/` folder |
| Streamlit won't start | `pip install streamlit --upgrade` |

## Authors

Group 3 Team - AAI6600 Fall 2025

## License

This project is for educational purposes as part of AAI6600 coursework.
