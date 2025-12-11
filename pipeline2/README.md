# Pipeline 2: CareConnect - Mental Health Support System

**AI-Powered Mental Health Classification & Facility Matching**

## What This Pipeline Does

CareConnect is an intelligent mental health support system that:
1. **Classifies mental health needs** from natural language using sentence-BERT embeddings
2. **Detects crisis situations** and provides immediate 988 hotline information
3. **Matches users with facilities** across 15,000+ mental health centers nationwide
4. **Scores facilities** across 7 dimensions using semantic similarity

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure OpenAI (optional)
cp config.json.example config.json
# Add your API key to config.json

# 3. Add facility data
cp your_facilities.csv data/all_facilities_scored.csv

# 4. Test
python tests/smoke_test.py

# 5. Run
python run_gui.py
```

## Architecture

**Classification:** 
- sentence-transformers (all-MiniLM-L6-v2, 384-dim embeddings)
- Logistic Regression with class weighting
- 14 mental health intent classes
- 91.3% accuracy

**Facility Scoring:**
- 7-dimensional semantic scoring
- Cosine similarity matching
- Weighted overall score (0-10)

## Project Structure

```
pipeline2/
├── README.md
├── requirements.txt
├── .gitignore
├── config.json.example
├── main.py (CLI)
├── run_gui.py (GUI launcher)
├── core/
│   ├── mental_health_classifier.py
│   ├── facility_scorer.py
│   ├── anti_hallucination.py
│   └── chatbot.py
├── gui/
│   └── streamlit_app.py
├── tests/
│   ├── smoke_test.py
│   ├── test_classifier.py
│   ├── test_scorer.py
│   └── generate_logs.py
├── data/
│   └── .gitkeep (add your CSV here)
├── logs/
└── conversation_logs/
```

## Usage

**GUI Mode:**
```bash
python run_gui.py
```

**CLI Mode:**
```bash
# Classify intent
python main.py classify "I'm feeling anxious"

# Search facilities
python main.py search --city Boston --state MA

# Generate conversation logs
python tests/generate_logs.py
```

## Expected Inputs & Outputs

**Input:** Natural language mental health concerns
- "I'm feeling anxious about exams"
- "I need help finding a therapist"
- "I don't want to live anymore" (triggers crisis protocol)

**Output:** Classification + Facilities
```json
{
  "intent": "anxiety_panic",
  "confidence": 0.87,
  "crisis_level": "STABLE",
  "facilities": [top 5 relevant facilities]
}
```

## Testing

```bash
python tests/smoke_test.py        # Quick validation
python tests/test_classifier.py    # Intent classification tests
python tests/generate_logs.py      # Generate 75+ conversation logs
```

## Dependencies

- sentence-transformers (embedding model)
- scikit-learn (classification)
- streamlit (GUI)
- openai (optional, for enhanced responses)
- pandas, numpy

## Setup Requirements

1. Python 3.8+
2. 4GB+ RAM
3. OpenAI API key (optional)
4. Facility CSV data

## Important Notes

⚠️ **Crisis Protocol**: System detects suicide risk and provides 988 hotline  
⚠️ **Not a replacement**: This is NOT professional mental health care  
⚠️ **Data Privacy**: Configure .gitignore properly (no API keys in git)

## Contact

**Course:** AAI 6600 Fall 2025, Northeastern University  
**Developer:** Nada Moursi  
**Instructor:** Professor A. Sinitskiy

---

**Version:** 1.0.0  
**Status:** ✅ Production Ready  
**Last Updated:** December 11, 2025
