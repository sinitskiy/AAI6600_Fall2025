# Mental Health Chatbot Pipeline

**AAI6600 Fall 2025 - Group 3**  
**Authors:** Subgroup B (Michael Zimmerman, Radhika Nori)

---

## 📋 Project Overview

A comprehensive mental health care facility routing system that provides end-to-end workflow from user conversation to facility recommendations. The system classifies mental health needs, routes to appropriate care groups, and matches users with affordable mental health facilities based on location and insurance status.

---

## 🏗️ System Architecture

```
User Input
    ↓
Conversation Interface (chatbot_interface.py)
    ↓
Classification (Mock/LLM) → Data Adapter (data_adapter.py)
    ↓
Routing (group2_router.py) → Group 3/4/Other Decision
    ↓
[If Group 3] → Collect Location/Insurance
    ↓
Facility Matching (facility_scorer.py)
    ↓
Recommendations (Top 5 facilities with scores)
```

---

## 📁 Project Structure

```
Basic framework/
├── chatbot_pipeline.py         # Main entry point - orchestrates full flow
├── chatbot_interface.py        # Terminal-based conversation interface
├── data_adapter.py             # Translation layer for multiple input sources
├── p1/
│   └── group2_router.py        # Routes categories to Group 3/4/Other
├── integrated/
│   ├── facility_scorer.py      # Facility matching and scoring system
│   └── main_workflow.py        # Original menu-based workflow
├── Group3_dataset/
│   └── all_facilities_scored.csv  # Pre-scored facility database
└── result_of_second_group/
    ├── test.py                 # Mock classification generator
    └── test.txt                # Sample classification output
```

---

## ✨ Key Features

### 1. **Flexible Input Sources**
- Mock classifier (testing)
- Real LLM integration (Subgroup A)
- TXT file input (Group 2 output)

### 2. **Data Adapter**
- Normalizes data from multiple sources
- Validates classification format
- Handles confidence normalization (85 → 0.85)
- Standardizes category names

### 3. **Intelligent Routing**
- 57 total categories across 3 groups
- Group 3 handles 27 affordable mental health categories
- Automatic handoff to Group 4 or other teams

### 4. **Location & Insurance Collection**
- Smart state name mapping (colorado → CO)
- Input validation and normalization
- Optional insurance provider tracking

### 5. **Facility Matching**
- Pre-scored CSV for fast lookup
- Location-based filtering (city, state)
- Scored recommendations (0-10 scale)
- Insurance and affordability information

---

## 🚀 Installation

### Prerequisites
```bash
# Python 3.8+
python3 --version

# Required packages
pip install pandas numpy
```

### Setup
```bash
# Clone repository
cd "/path/to/AAI6600_Group3/Basic framework"

# Verify directory structure
ls -la
# Should see: chatbot_pipeline.py, data_adapter.py, p1/, integrated/, etc.

# Ensure pre-scored CSV exists
ls Group3_dataset/all_facilities_scored.csv
```

---

## 🎯 Usage

### Option 1: Run Complete Pipeline (Recommended)

```bash
python3 chatbot_pipeline.py
```

**What happens:**
1. Mock conversation is classified
2. Router determines if Group 3 handles it
3. If yes → collects location/insurance from user
4. Searches facilities and displays top 5 recommendations

**Example interaction:**
```
What city are you in? Denver
What state are you in? CO
What is your ZIP code? (4 or 5 digits, press Enter to skip) 80204
Do you have health insurance? (yes/no) yes
If yes, what's your insurance provider? Medicaid
Choose output view - (1) Simple (non-technical)  (2) JSON (technical) [default 1]: 1

[Shows 5 facilities in Denver, CO with scores]
```

### Option 2: Test Data Adapter

```bash
python3 data_adapter.py
```

**Output:** Runs 7 test cases validating adapter functionality

### Option 3: Interactive Chatbot Interface

```bash
python3 chatbot_interface.py
```

**What happens:**
- Terminal-based chat interface
- Full conversation history with scrolling
- Mainframe-style aesthetic
- Type 'quit' to exit

---

## 🔧 Integration Guide for Subgroup A (LLM Team)

### Required LLM Output Format

Your LLM classifier must return a dictionary with these fields:

```python
{
    'category': 'Mental health',  # One of 57 categories
    'confidence': 0.85,           # 0.0-1.0 or 0-100 (both work)
    'user_input': 'User said they feel anxious and need help'
}
```

### Flexible Format Support

The adapter handles multiple format variations:

**Option A:**
```python
{'category': 'depression', 'confidence': 85, 'symptoms': '...'}
```

**Option B:**
```python
{'intent': 'crisis', 'certainty': 0.95, 'user_text': '...'}
```

**Option C:**
```python
{'classification': 'anxiety', 'score': 78, 'input': '...'}
```

### Integration Steps

1. **Create your LLM function:**
```python
def classify_conversation(conversation_history):
    # Your LLM logic here
    return {
        'category': 'Mental health',
        'confidence': 0.85,
        'user_input': 'summary of conversation'
    }
```

2. **Update chatbot_pipeline.py (line ~196):**
```python
# Replace this:
from data_adapter import adapt_mock_output
raw_classification = mock_classify_conversation(mock_conversation)
classification = adapt_mock_output(raw_classification)

# With this:
from data_adapter import adapt_llm_output
from your_module import classify_conversation
raw_classification = classify_conversation(mock_conversation)
classification = adapt_llm_output(raw_classification)
```

3. **Test integration:**
```python
from data_adapter import validate_llm_integration
validate_llm_integration(your_llm_function)
```

---

## 📊 System Flow Example

### Complete End-to-End Flow

```
[Step 1] Mock Conversation
User: "I've been feeling really anxious and need help"
User: "I have panic attacks and trouble sleeping"

[Step 2] Classification
Mock classifier → {'category': 'mental health', 'confidence': 0.85}
Adapter normalizes → {'category': 'Mental health', 'confidence': 0.85}

[Step 3] Routing
Router: "Group 3 handles Mental health category" ✓

[Step 4] Information Collection
City: Denver
State: CO (normalized from 'co')
Insurance: Yes
Provider: Medicaid

[Step 5] Facility Matching
Searching all_facilities_scored.csv...
Found 5 facilities in Denver, CO
Sorted by overall_care_needs_score

[Results]
1. Jefferson Center - West Colfax (Score: 7.6/10)
2. Colorado Mental Health Institute (Score: 7.6/10)
3. SOL Mental Health - Cherry Creek (Score: 7.4/10)
4. Jefferson Center - Union Office (Score: 7.3/10)
5. Denver Health Authority (Score: 7.3/10)
```

---

## 🧪 Testing

### Test Adapter Functionality
```bash
python3 data_adapter.py
```

**Expected output:**
```
✓ [Test 1] Mock Classifier Output
✓ [Test 2] LLM Output - Format A
✓ [Test 3] LLM Output - Format B
✓ [Test 4] TXT File Output
✓ [Test 5] Auto-Detection
✓ [Test 6] Validation - Missing Field
✓ [Test 7] Confidence Normalization
ALL TESTS COMPLETE
```

### Test Different Scenarios
```bash
# Test with Charlotte, NC
python3 chatbot_pipeline.py
# Enter: Charlotte, NC, yes, Medicare

# Test with New York, NY
python3 chatbot_pipeline.py
# Enter: New York, NY, no

# Test with full state name
python3 chatbot_pipeline.py
# Enter: Boston, Massachusetts, yes
```

---

## 📦 Components Deep Dive

### 1. chatbot_pipeline.py
**Purpose:** Main orchestrator for entire flow

**Key Functions:**
- `mock_classify_conversation()` - Temporary classifier (replace with LLM)
- `collect_additional_info()` - Gathers location/insurance
- `call_facility_matcher()` - Searches pre-scored CSV
- `run_pipeline()` - Master orchestration function

### 2. data_adapter.py
**Purpose:** Translation layer between input sources and router

**Key Functions:**
- `adapt_mock_output()` - Handles mock classifier format
- `adapt_llm_output()` - Handles LLM formats (flexible)
- `adapt_txt_output()` - Handles Group 2 TXT files
- `normalize_category()` - Standardizes category names
- `normalize_confidence()` - Converts percentages to decimals

### 3. chatbot_interface.py
**Purpose:** Terminal-based conversational UI

**Features:**
- Mainframe-style aesthetic
- Full conversation history with scrolling
- Timestamp for each message
- Clear visual distinction (>> user, << bot)
- Mock conversation flow for testing

### 4. group2_router.py (p1/)
**Purpose:** Routes categories to appropriate groups

**Categories:**
- **Group 3 (27 categories):** Affordable mental health care
- **Group 4 (6 categories):** Peer support groups
- **Other (24 categories):** Academic, career, campus services

---

## 🎯 Group 3 Categories (27 Total)

The router identifies these as Group 3 responsibilities:

```
Mental health, Mental health support, Counseling, 
Counseling support, Psychiatrist, Crisis counseling, 
Crisis line, Crisis services, Trauma counseling, 
Grief counseling, Group therapy, Emotional regulation group, 
Skills group, LGBTQ+ counseling, Cultural counseling, 
Cultural adjustment counseling, Accessibility counseling, 
Self-care, Self-help, Wellness support, Health, 
Virtual counseling, Directory of free mental health providers, 
Specialist, Parenting support, Case management
```

---

## 🔄 State Name Mapping

The system intelligently handles both 2-letter codes and full state names:

```python
# All accepted formats:
"NC" or "North Carolina" → NC
"co" or "Colorado" → CO
"massachusetts" or "MA" → MA
```

**Supported:** All 50 US states with full name or 2-letter code

---

## 💾 Data Files

### all_facilities_scored.csv
**Location:** `Group3_dataset/all_facilities_scored.csv`

**Key Fields:**
- `name` - Facility name
- `street`, `city`, `state`, `zipcode` - Location
- `phone` - Contact number
- `overall_care_needs_score` - Primary ranking (0-10)
- `affordability_score` - Cost-related score
- `crisis_care_score` - Emergency care capability
- `accepts_medicaid`, `accepts_medicare` - Insurance flags
- `affordable_flag`, `affordability_tier` - Cost categories

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Ensure you're in correct directory
cd "/Volumes/Transcend/AAI6600_Group3/Basic framework"
pwd  # Should show: .../Basic framework

# Check Python path
python3 -c "import sys; print(sys.path)"
```

### Missing CSV File
```
⚠️  No pre-scored dataset found at Group3_dataset/all_facilities_scored.csv
```
**Solution:** Ensure CSV exists or run facility_scorer.py to generate it

### Adapter Validation Errors
```
ValueError: Invalid mock data: Missing required fields: confidence
```
**Solution:** Check your classifier returns all required fields:
- `category` (string)
- `confidence` (number, 0-1 or 0-100)
- `user_input` (string)

### State Not Recognized
```
I couldn't recognize that state. Please enter 2-letter code or full state name
```
**Solution:** Use 2-letter code (NC, CO, MA) or full name (North Carolina, Colorado, Massachusetts)

---

## 🚦 Status & Next Steps

### ✅ Completed
- [x] End-to-end pipeline architecture
- [x] Mock classifier for testing
- [x] Data adapter with flexible input support
- [x] Router integration (Group 3/4/Other)
- [x] Location & insurance collection
- [x] Facility matching with pre-scored CSV
- [x] Clean output formatting (simple & JSON)
- [x] State name normalization
- [x] Comprehensive test suite

### 🔄 In Progress
- [ ] LLM integration (Subgroup A)
- [ ] Real chatbot conversation interface connection
- [ ] Multilingual support (Spanish, Portuguese, Chinese)

### 📋 Future Enhancements
- [ ] GUI interface
- [ ] Real-time LLM conversation
- [ ] Enhanced facility filtering (distance, ratings)
- [ ] User feedback collection
- [ ] Session history tracking

---

## 👥 Team Structure

### Subgroup A: Conversation & LLM Integration
- Janet Garcia
- Sharan Sandhu
- Sneha Paul

**Responsibilities:**
- LLM integration with Gemini/Groq API
- Natural language processing
- Multilingual support
- Prompt engineering

### Subgroup B: Pipeline Integration & Testing
- Michael Zimmerman (Group Leader)
- Radhika Nori

**Responsibilities:**
- Pipeline architecture
- Data adapter implementation
- Router integration
- Facility matching connection
- Testing & validation

---

## 📞 Contact & Support

**Project Repository:** AAI6600_Group3/Basic framework  
**Course:** AAI6600 Fall 2025  
**Institution:** [University Name]

For questions or issues:
1. Check this README first
2. Review code comments in respective files
3. Run test suites to validate setup
4. Contact team leads for integration questions

---

## 📄 License

Academic project for AAI6600 Fall 2025. All rights reserved.

---

## 🙏 Acknowledgments

- **Group 2:** Classification system and routing logic
- **Group 3 (Original):** Facility scoring and dataset preparation
- **Group 4:** Peer support categories integration
- **Professor:** Project guidance and requirements

---

**Last Updated:** November 3, 2025  
**Version:** 1.0  
**Status:** Production Ready for Demo
