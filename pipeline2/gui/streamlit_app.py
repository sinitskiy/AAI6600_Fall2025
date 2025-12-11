#!/usr/bin/env python3
"""
Pipeline 2 - Streamlit GUI Application

Beautiful web interface for:
1. Mental Health Intent Classification
2. Facility Semantic Search
3. Analytics Dashboard
4. About Pipeline 2
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from core.mental_health_classifier import MentalHealthClassifier
from core.facility_scorer import FacilityScorer
from core.anti_hallucination import AntiHallucinationValidator

# Page configuration
st.set_page_config(
    page_title="Pipeline 2 - Mental Health AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .crisis-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
    }
    .facility-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'scorer' not in st.session_state:
    st.session_state.scorer = None
if 'validator' not in st.session_state:
    st.session_state.validator = None
if 'facilities_df' not in st.session_state:
    st.session_state.facilities_df = None


@st.cache_resource
def load_classifier():
    """Load mental health classifier (cached)"""
    return MentalHealthClassifier()


@st.cache_resource
def load_scorer():
    """Load facility scorer (cached)"""
    return FacilityScorer()


@st.cache_resource
def load_validator():
    """Load anti-hallucination validator (cached)"""
    return AntiHallucinationValidator()


@st.cache_data
def load_facilities_data():
    """Load facilities data (cached)"""
    data_dir = parent_dir / "data"
    
    # Try to find scored facilities
    possible_files = [
        data_dir / "all_facilities_scored.csv",
        parent_dir.parent / "Group3_dataset" / "all_facilities_scored.csv",
        data_dir / "facilities_scored.csv"
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, low_memory=False)
                return df
            except Exception as e:
                st.warning(f"Could not load {file_path.name}: {e}")
    
    return None


# Header
st.markdown('<div class="main-header">🧠 Pipeline 2: Mental Health AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Embedding-Based Classification & Facility Matching</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Northeastern_seal.svg/1200px-Northeastern_seal.svg.png", width=100)
    st.markdown("### Pipeline 2")
    st.markdown("**AAI 6600 Fall 2025**")
    st.markdown("Northeastern University")
    st.divider()
    
    st.markdown("### 📊 System Status")
    
    # Load models
    try:
        if st.session_state.classifier is None:
            with st.spinner("Loading classifier..."):
                st.session_state.classifier = load_classifier()
        st.success("✓ Classifier ready")
    except Exception as e:
        st.error(f"✗ Classifier failed: {e}")
    
    try:
        if st.session_state.scorer is None:
            with st.spinner("Loading scorer..."):
                st.session_state.scorer = load_scorer()
        st.success("✓ Scorer ready")
    except Exception as e:
        st.error(f"✗ Scorer failed: {e}")
    
    try:
        if st.session_state.validator is None:
            st.session_state.validator = load_validator()
        st.success("✓ Validator ready")
    except Exception as e:
        st.error(f"✗ Validator failed: {e}")
    
    # Load data
    if st.session_state.facilities_df is None:
        st.session_state.facilities_df = load_facilities_data()
    
    if st.session_state.facilities_df is not None:
        st.success(f"✓ {len(st.session_state.facilities_df):,} facilities loaded")
    else:
        st.warning("⚠ Using mock data")


# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🧠 Mental Health Classifier",
    "🏥 Facility Search",
    "📊 Analytics",
    "📚 About Pipeline 2"
])


# ============================================================================
# TAB 1: MENTAL HEALTH CLASSIFIER
# ============================================================================

with tab1:
    st.header("🧠 Mental Health Intent Classification")
    st.markdown("Test the ML classifier (TF-IDF + SVM) on mental health text")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter User Input")
        user_input = st.text_area(
            "What would the user say?",
            placeholder="Example: I'm feeling really anxious about my exams...",
            height=120
        )
        
        classify_button = st.button("🔍 Classify Intent", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### Quick Examples")
        st.markdown("**Copy and paste these:**")
        st.code("I'm having panic attacks and can't stop worrying", language=None)
        st.code("I feel so sad and hopeless", language=None)
        st.code("I don't want to live anymore", language=None)
        st.code("I'm so overwhelmed with school", language=None)
    
    if classify_button and user_input:
        if st.session_state.classifier is None:
            st.error("Classifier not loaded. Please refresh the page.")
        else:
            with st.spinner("Classifying..."):
                result = st.session_state.classifier.classify_with_response(user_input)
            
            st.divider()
            
            # Display results
            if result['is_crisis']:
                st.markdown(f"""
                <div class="crisis-alert">
                    <h2>🚨 CRISIS DETECTED</h2>
                    <p><strong>Intent:</strong> {result['intent'].upper()}</p>
                    <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.error("**EMERGENCY RESOURCES:**")
                st.markdown(result['response'])
            else:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Detected Intent", result['intent'].title())
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                
                with col3:
                    crisis_status = "🚨 Yes" if result['is_crisis'] else "✅ No"
                    st.metric("Crisis?", crisis_status)
                
                st.markdown("### 💬 Suggested Response")
                st.info(result['response'])
            
            # Show technical details
            with st.expander("🔬 Technical Details"):
                st.json(result)


# ============================================================================
# TAB 2: FACILITY SEARCH
# ============================================================================

with tab2:
    st.header("🏥 Mental Health Facility Search")
    st.markdown("Semantic similarity-based facility matching (BERT embeddings)")
    
    # Search filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city_input = st.text_input("🏙️ City", placeholder="e.g., Boston")
    
    with col2:
        state_input = st.text_input("🗺️ State", placeholder="e.g., MA")
    
    with col3:
        top_n = st.slider("📊 Number of Results", 1, 10, 5)
    
    # Additional filters
    with st.expander("⚙️ Advanced Filters"):
        col1, col2 = st.columns(2)
        
        with col1:
            needs_filter = st.multiselect(
                "Special Needs",
                ["crisis", "affordable", "child", "trauma", "addiction", "anxiety", "depression"],
                help="Filter facilities by specific capabilities"
            )
        
        with col2:
            insurance_type = st.selectbox(
                "Insurance Type",
                ["Any", "Medicaid", "Medicare", "Private Insurance", "No Insurance"],
                help="Filter by insurance acceptance"
            )
    
    search_button = st.button("🔍 Search Facilities", type="primary", use_container_width=True)
    
    if search_button:
        if st.session_state.facilities_df is None:
            st.warning("⚠️ No facility data loaded. Using mock data...")
            
            # Mock data for demo
            mock_facilities = pd.DataFrame({
                'name': [
                    'Community Mental Health Center',
                    'City Counseling Services',
                    'Affordable Care Clinic',
                    'Crisis Intervention Center',
                    'Family Therapy Associates'
                ],
                'street': ['123 Main St', '456 Elm Ave', '789 Oak Rd', '321 Pine St', '654 Maple Dr'],
                'city': [city_input or 'Boston'] * 5,
                'state': [state_input or 'MA'] * 5,
                'zipcode': ['02115', '02116', '02117', '02118', '02119'],
                'phone': ['(617) 555-0001', '(617) 555-0002', '(617) 555-0003', '(617) 555-0004', '(617) 555-0005'],
                'overall_care_needs_score': [8.5, 7.8, 8.2, 9.1, 7.5],
                'affordability_score': [9.0, 8.5, 9.5, 7.0, 7.5],
                'crisis_care_score': [7.5, 6.0, 6.5, 10.0, 5.5],
                'accessibility_score': [8.0, 8.5, 7.5, 8.5, 7.0]
            })
            
            results = mock_facilities.head(top_n)
        
        else:
            # Real search
            with st.spinner("Searching facilities..."):
                if st.session_state.scorer is None:
                    st.session_state.scorer = load_scorer()
                
                results = st.session_state.scorer.get_top_facilities(
                    st.session_state.facilities_df,
                    n=top_n,
                    city=city_input if city_input else None,
                    state=state_input if state_input else None,
                    needs=needs_filter if needs_filter else None
                )
        
        st.divider()
        
        # Display results
        if len(results) == 0:
            st.warning("No facilities found matching your criteria. Try expanding your search.")
        else:
            st.success(f"Found {len(results)} facilities")
            
            # Display each facility
            for idx, (_, facility) in enumerate(results.iterrows(), 1):
                with st.container():
                    st.markdown(f"""
                    <div class="facility-card">
                        <h3>{idx}. {facility.get('name', 'Unknown Facility')}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **📍 Address:** {facility.get('street', 'N/A')}  
                        **🏙️ Location:** {facility.get('city', 'N/A')}, {facility.get('state', 'N/A')} {facility.get('zipcode', '')}  
                        **📞 Phone:** {facility.get('phone', 'N/A')}
                        """)
                    
                    with col2:
                        overall_score = facility.get('overall_care_needs_score', 0)
                        st.metric("Overall Score", f"{overall_score:.1f}/10")
                    
                    # Score breakdown
                    with st.expander("📊 Score Breakdown"):
                        score_cols = st.columns(5)
                        
                        scores = {
                            '💰 Affordability': facility.get('affordability_score', 0),
                            '🚨 Crisis Care': facility.get('crisis_care_score', 0),
                            '🚪 Accessibility': facility.get('accessibility_score', 0),
                            '🎯 Specialization': facility.get('specialization_score', 0),
                            '🏘️ Community': facility.get('community_integration_score', 0)
                        }
                        
                        for col, (label, score) in zip(score_cols, scores.items()):
                            with col:
                                st.metric(label, f"{score:.1f}")
                    
                    st.divider()


# ============================================================================
# TAB 3: ANALYTICS DASHBOARD
# ============================================================================

with tab3:
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.facilities_df is not None:
        df = st.session_state.facilities_df
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Facilities", f"{len(df):,}")
        
        with col2:
            states_count = df['state'].nunique()
            st.metric("States Covered", states_count)
        
        with col3:
            avg_score = df['overall_care_needs_score'].mean()
            st.metric("Avg Score", f"{avg_score:.2f}/10")
        
        with col4:
            high_score = (df['overall_care_needs_score'] >= 7.5).sum()
            st.metric("High Score (7.5+)", f"{high_score:,}")
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📍 Facilities by State (Top 10)")
            state_counts = df['state'].value_counts().head(10)
            fig = px.bar(
                x=state_counts.index,
                y=state_counts.values,
                labels={'x': 'State', 'y': 'Number of Facilities'},
                color=state_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Overall Score Distribution")
            fig = px.histogram(
                df,
                x='overall_care_needs_score',
                nbins=20,
                labels={'overall_care_needs_score': 'Overall Score'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Dimension comparison
        st.markdown("### 🎯 Average Scores by Dimension")
        
        dimensions = {
            'Affordability': 'affordability_score',
            'Crisis Care': 'crisis_care_score',
            'Accessibility': 'accessibility_score',
            'Specialization': 'specialization_score',
            'Community Integration': 'community_integration_score'
        }
        
        avg_scores = {}
        for dim_name, col_name in dimensions.items():
            if col_name in df.columns:
                avg_scores[dim_name] = df[col_name].mean()
        
        if avg_scores:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(avg_scores.keys()),
                    y=list(avg_scores.values()),
                    marker_color='#1f77b4',
                    text=[f"{v:.2f}" for v in avg_scores.values()],
                    textposition='auto'
                )
            ])
            fig.update_layout(
                yaxis_title="Average Score (0-10)",
                xaxis_title="Dimension",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("📋 View Raw Data"):
            st.dataframe(df.head(100), use_container_width=True)
    
    else:
        st.info("📂 No facility data loaded. Add CSV files to the data/ folder.")
        st.markdown("""
        To use real facility data:
        1. Copy `all_facilities_scored.csv` to `pipeline2/data/`
        2. Refresh this page
        """)


# ============================================================================
# TAB 4: ABOUT PIPELINE 2
# ============================================================================

with tab4:
    st.header("📚 About Pipeline 2")
    
    st.markdown("""
    ## 🎯 What is Pipeline 2?
    
    Pipeline 2 is a **hybrid machine learning system** that combines:
    - **Sentence-transformer embeddings** (all-MiniLM-L6-v2) for semantic understanding
    - **Logistic Regression classifier** for mental health need detection  
    - **Semantic facility scoring** with 5-dimensional evaluation
    - **Anti-hallucination validation** to prevent false recommendations
    
    ### ✨ Key Features
    
    1. **🧠 Intent Classification**
       - TF-IDF + SVM classifier
       - 8 intent categories (crisis, anxiety, depression, etc.)
       - ~90% accuracy on test set
    
    2. **🏥 Facility Scoring**
       - 11,000+ facilities from multiple sources
       - 5-dimensional semantic scoring
       - BERT-based similarity (0-10 scale)
    
    3. **✅ Anti-Hallucination Validation**
       - Detects negative evidence (cash-only services)
       - Validates data completeness
       - Reliability scoring (0-100)
    
    ### 📊 Performance Metrics
    
    | Metric | Value |
    |--------|-------|
    | Intent Classification Accuracy | 90% |
    | Crisis Detection Recall | 95% |
    | Facility Coverage | 11,000+ |
    | Average Scoring Time | 2-3 min (1000 facilities) |
    
    ### 🔬 Research Paper Context
    
    This pipeline is part of the research paper:
    
    **"Advanced AI vs Manual Software Development in Mental Health Text Analysis App"**
    
    **Key Findings:**
    - ✅ Manual Pipeline 2 achieved 90% detection accuracy
    - ✅ 6% crisis omission rate (vs 43% for AI-generated systems)
    - ✅ Complete end-to-end functionality without AI frameworks
    
    ### 👥 Team
    
    **AAI 6600 Fall 2025 - Northeastern University**
    
    ### 📖 How to Use
    
    1. **Classifier Tab**: Test mental health intent detection
    2. **Facility Search Tab**: Find facilities by location and needs
    3. **Analytics Tab**: View pipeline statistics and distributions
    
    ### 🛠️ Technical Stack
    
    - **Python 3.8+**
    - **sentence-transformers** - BERT embeddings
    - **scikit-learn** - ML classification
    - **Streamlit** - Web interface
    - **Plotly** - Interactive visualizations
    """)
    
    st.divider()
    
    st.markdown("### 📞 Emergency Resources")
    st.error("""
    **If you or someone you know is in crisis:**
    - 🆘 Call/Text: **988** (Suicide & Crisis Lifeline)
    - 💬 Text: **HOME to 741741** (Crisis Text Line)
    - 🚨 Emergency: **911**
    """)


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Pipeline 2</strong> - Embedding-Based Mental Health Classification</p>
    <p>AAI 6600 Fall 2025 | Northeastern University</p>
    <p>⚠️ This is a research project. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)
