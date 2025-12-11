#!/usr/bin/env python3
"""
Facility Scoring Module
Uses sentence-transformers for semantic similarity scoring

Based on: all-MiniLM-L6-v2 model (384-dimensional embeddings)
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FacilityScorer:
    """
    Facility Scorer - Semantic Similarity Based
    
    Uses BERT embeddings to score facilities across 5 dimensions:
    1. Affordability
    2. Crisis care
    3. Accessibility
    4. Specialization
    5. Community integration
    """
    
    # Targeted questions for each dimension
    TARGETED_QUESTIONS = {
        'affordability': """Does this mental health facility accept Medicaid, Medicare, or offer sliding scale fees, free services, or low-cost counseling?""",
        
        'crisis_care': """Does this facility provide crisis intervention services, emergency mental health care, crisis hotline, suicide prevention, 24/7 urgent psychiatric services?""",
        
        'accessibility': """Is this facility easily accessible with walk-in services, multiple locations, convenient hours, telehealth options, or community-based outreach?""",
        
        'specialization': """Does this facility specialize in specific mental health services such as child psychology, trauma treatment, PTSD care, addiction treatment, or eating disorders?""",
        
        'community_integration': """Is this facility integrated with the community, serving as a neighborhood mental health center, community behavioral health organization, or nonprofit focused on underserved populations?"""
    }
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize scorer
        
        Args:
            model_name: Sentence Transformer model name
        """
        print("Loading facility scorer...")
        
        try:
            self.model = SentenceTransformer(model_name)
            print("✓ Semantic model loaded successfully")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            raise
        
        # Pre-compute question embeddings
        print("Pre-computing question embeddings...")
        self.question_embeddings = {}
        for key, question in self.TARGETED_QUESTIONS.items():
            self.question_embeddings[key] = self.model.encode(
                question.strip(),
                convert_to_tensor=False,
                show_progress_bar=False
            )
        print("✓ Scorer initialization complete\n")
    
    def create_clean_text(self, row):
        """Create clean text for semantic analysis"""
        parts = [
            str(row.get('name', '')),
            str(row.get('street', '')),
            str(row.get('city', '')),
            str(row.get('state', ''))
        ]
        
        clean_parts = [
            p.lower() for p in parts
            if p and str(p).strip() and str(p) != 'nan'
        ]
        
        return ' '.join(clean_parts)
    
    def calculate_cosine_similarity(self, text, question_embedding):
        """Calculate cosine similarity between text and question"""
        if pd.isna(text) or not str(text).strip():
            return 0.0
        
        try:
            text_embedding = self.model.encode(
                str(text),
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            similarity = np.dot(question_embedding, text_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(text_embedding) + 1e-8
            )
            
            return float(similarity)
        except:
            return 0.0
    
    def similarity_to_score(self, similarity):
        """
        Convert similarity (0-1) to score (0-10)
        
        Mapping:
        [0.00, 0.20) → [0, 2)
        [0.20, 0.30) → [2, 4)
        [0.30, 0.40) → [4, 6)
        [0.40, 0.50) → [6, 8)
        [0.50, 1.00] → [8, 10]
        """
        similarity = max(0.0, similarity)
        
        if similarity < 0.20:
            score = similarity / 0.20 * 2
        elif similarity < 0.30:
            score = 2 + (similarity - 0.20) / 0.10 * 2
        elif similarity < 0.40:
            score = 4 + (similarity - 0.30) / 0.10 * 2
        elif similarity < 0.40:
            score = 6 + (similarity - 0.40) / 0.10 * 2
        else:
            score = 8 + (similarity - 0.50) / 0.50 * 2
        
        return max(0.0, min(10.0, score))
    
    def score_facilities(self, df, text_column='clean_text'):
        """
        Score facility dataframe
        
        Args:
            df: Facility dataframe
            text_column: Text column name for scoring
        
        Returns:
            Scored dataframe with similarity and score columns
        """
        print(f"\nScoring {len(df)} facilities...")
        
        # Generate clean_text if not present
        if text_column not in df.columns:
            print("   Generating clean_text...")
            df[text_column] = df.apply(self.create_clean_text, axis=1)
        
        # Calculate similarity for each dimension
        print("   Calculating multi-dimensional similarities...")
        
        for key in self.TARGETED_QUESTIONS.keys():
            df[f'{key}_similarity'] = 0.0
        
        for idx in tqdm(range(len(df)), desc="   Processing"):
            clean_text = str(df.iloc[idx][text_column])
            
            for key, question_emb in self.question_embeddings.items():
                similarity = self.calculate_cosine_similarity(clean_text, question_emb)
                df.at[df.index[idx], f'{key}_similarity'] = similarity
        
        # Convert to 0-10 scores
        print("   Converting to scores...")
        
        score_cols = []
        for key in self.TARGETED_QUESTIONS.keys():
            sim_col = f'{key}_similarity'
            score_col = f'{key}_score'
            score_cols.append(score_col)
            
            df[score_col] = df[sim_col].apply(self.similarity_to_score)
        
        # Calculate overall score
        df['overall_care_needs_score'] = df[score_cols].mean(axis=1)
        
        print("✓ Scoring complete\n")
        
        return df
    
    def get_top_facilities(self, df, n=5, city=None, state=None, needs=None):
        """
        Get top N facilities with optional filtering
        
        Args:
            df: Scored facility dataframe
            n: Number of facilities to return
            city: City filter (optional)
            state: State filter (optional)
            needs: List of special needs (optional)
        
        Returns:
            Top N facilities
        """
        filtered_df = df.copy()
        
        # Geographic filtering
        if state:
            filtered_df = filtered_df[
                filtered_df['state'].str.upper() == state.upper()
            ]
        
        if city:
            filtered_df = filtered_df[
                filtered_df['city'].str.contains(city, case=False, na=False)
            ]
        
        # Sort by overall score
        if 'overall_care_needs_score' in filtered_df.columns:
            filtered_df = filtered_df.sort_values(
                'overall_care_needs_score',
                ascending=False
            )
        
        return filtered_df.head(n)


# For testing
if __name__ == "__main__":
    print("=" * 70)
    print("Facility Scorer - Test")
    print("=" * 70)
    
    # Create test dataframe
    test_data = {
        'name': ['Community Mental Health Center', 'Private Practice', 'City Hospital'],
        'street': ['123 Main St', '456 Oak Ave', '789 Hospital Rd'],
        'city': ['Boston', 'Boston', 'Boston'],
        'state': ['MA', 'MA', 'MA']
    }
    
    df = pd.DataFrame(test_data)
    
    # Score facilities
    scorer = FacilityScorer()
    df_scored = scorer.score_facilities(df)
    
    # Display results
    print("\n✓ Test complete!")
    print("\nScored facilities:")
    print(df_scored[['name', 'overall_care_needs_score']].to_string())
    
