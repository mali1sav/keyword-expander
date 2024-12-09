import os
import time
import pandas as pd
import streamlit as st
from serpapi import Client
from dotenv import load_dotenv
from typing import List, Dict, Set
import numpy as np
from openai import OpenAI
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# Load environment variables
load_dotenv()

# Check for API keys at startup
SERP_API_KEY = os.getenv("SERP_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SERP_API_KEY:
    st.error("Please ensure SERP_API_KEY is set in your .env file")
    st.stop()

if not OPENAI_API_KEY:
    st.error("Please ensure OPENAI_API_KEY is set in your .env file")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize SerpAPI client
serp_client = Client(api_key=SERP_API_KEY)

# Constants
MAX_DEPTH = 2
RATE_LIMIT_DELAY = 1  # Delay in seconds between API calls

# Initialize session state
if 'expanded_keywords' not in st.session_state:
    st.session_state.expanded_keywords = None

def fetch_google_keywords(keyword: str, country: str) -> Dict[str, List[str]]:
    """Fetch keywords from Google"""
    params = {
        "engine": "google",
        "q": keyword,
        "google_domain": "google.co.jp" if country == "jp" else "google.co.th",
        "gl": country,
        "hl": "ja" if country == "jp" else "th"
    }
    
    results = {
        "related_keywords": [],
        "people_also_ask": [],
        "autocomplete": []
    }
    
    try:
        # Fetch main search results
        search = serp_client.search(params)
        
        # Debug information
        st.write(f"Found {len(search.get('related_searches', []))} related searches")
        st.write(f"Found {len(search.get('related_questions', []))} related questions")
        
        # Extract related searches
        if 'related_searches' in search:
            results["related_keywords"] = [item["query"] for item in search["related_searches"]]
        
        # Extract people also ask
        if 'related_questions' in search:
            results["people_also_ask"] = [item["question"] for item in search["related_questions"]]
        
        try:
            # Fetch autocomplete suggestions separately
            autocomplete_params = params.copy()
            autocomplete_params["engine"] = "google_autocomplete"
            autocomplete_search = serp_client.search(autocomplete_params)
            
            if "suggestions" in autocomplete_search:
                suggestions = []
                for item in autocomplete_search["suggestions"]:
                    if isinstance(item, dict) and "suggestion" in item:
                        suggestions.append(item["suggestion"])
                results["autocomplete"] = suggestions
                st.write(f"Found {len(suggestions)} autocomplete suggestions")
            
        except Exception as e:
            st.write(f"Debug: Autocomplete fetch failed: {str(e)}")
            pass
        
    except Exception as e:
        st.error(f"Error fetching Google data for keyword '{keyword}': {str(e)}")
    
    return results

def fetch_yahoo_jp_keywords(keyword: str) -> Dict[str, List[str]]:
    """Fetch keywords from Yahoo Japan"""
    params = {
        "engine": "yahoo_jp",
        "q": keyword,
    }
    
    results = {
        "related_search": [],
        "related_questions": [],
        "related_topics": [],
        "trending_searches": []
    }
    
    try:
        search = serp_client.search(params)
        
        # Extract different types of related content
        if 'related_searches' in search:
            results["related_search"] = [item["query"] for item in search["related_searches"]]
            
        if 'related_questions' in search:
            results["related_questions"] = [item["question"] for item in search["related_questions"]]
            
        if 'related_topics' in search:
            results["related_topics"] = [item["topic"] for item in search["related_topics"]]
            
        if 'trending_searches' in search:
            results["trending_searches"] = [item["query"] for item in search["trending_searches"]]
        
    except Exception as e:
        st.error(f"Error fetching Yahoo Japan data for keyword '{keyword}': {str(e)}")
    
    return results

def expand_keywords(seed_keywords: List[str], depth: int, country: str, search_engines: List[str]) -> Set[str]:
    """Expand keywords using selected search engines"""
    all_keywords = set(seed_keywords)
    keywords_to_process = seed_keywords.copy()
    
    progress_bar = st.progress(0)
    total_iterations = depth
    
    for current_depth in range(depth):
        new_keywords = []
        for keyword in keywords_to_process:
            if "Google" in search_engines:
                google_results = fetch_google_keywords(keyword, country)
                new_keywords.extend(google_results["related_keywords"])
                new_keywords.extend(google_results["people_also_ask"])
                new_keywords.extend(google_results["autocomplete"])
            
            if "Yahoo Japan" in search_engines and country == "jp":
                yahoo_results = fetch_yahoo_jp_keywords(keyword)
                new_keywords.extend(yahoo_results["related_search"])
                new_keywords.extend(yahoo_results["related_questions"])
                new_keywords.extend(yahoo_results["related_topics"])
                new_keywords.extend(yahoo_results["trending_searches"])
            
            time.sleep(RATE_LIMIT_DELAY)
        
        keywords_to_process = list(set(new_keywords) - all_keywords)
        all_keywords.update(new_keywords)
        progress_bar.progress((current_depth + 1) / total_iterations)
    
    return all_keywords

def get_embedding(text):
    """Get embedding for a single text using OpenAI API"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def create_visualization(texts):
    """Create an interactive visualization of text similarities"""
    if not texts:
        st.warning("No input texts provided.")
        return
    
    with st.spinner("Getting embeddings..."):
        embeddings = []
        processed_texts = texts.copy()  # Use texts as is, no processing needed
        
        for text in processed_texts:
            embedding = get_embedding(text)
            embeddings.append(embedding)
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)
    
    with st.spinner("Reducing dimensions..."):
        # Reduce dimensions using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings_array)
    
    # Calculate similarity scores based on distances to nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(3, len(processed_texts))).fit(embeddings_array)
    distances, _ = nbrs.kneighbors(embeddings_array)
    similarity_scores = 1 / (1 + np.mean(distances, axis=1))
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'text': processed_texts,
        'short_text': [text[:30] + "..." if len(text) > 30 else text for text in processed_texts]
    })
    
    # Create custom hover text
    hover_text = [f"Keyword: {text}" for text in processed_texts]
    
    # Create figure with custom layout
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers+text',
        text=df['short_text'],
        hovertext=hover_text,
        hoverinfo='text',
        textposition='top center',
        marker=dict(
            size=10,
            color=similarity_scores,
            colorscale='Viridis',
            opacity=0.7,
            showscale=True,
            colorbar=dict(
                title='Similarity Score',
                tickformat='.2f'
            )
        ),
        textfont=dict(size=10)
    ))
    
    # Update layout for better readability and full screen
    fig.update_layout(
        title={
            'text': 'Keyword Similarity Visualization',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis=dict(
            title='t-SNE dimension 1',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211,211,211,0.3)'
        ),
        yaxis=dict(
            title='t-SNE dimension 2',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211,211,211,0.3)'
        ),
        plot_bgcolor='white',
        width=None,  # Allow full width
        height=900,  # Increased height
        showlegend=False,
        hovermode='closest'
    )
    
    # Add zoom and pan buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Reset View",
                         method="relayout",
                         args=[{"xaxis.range": [None, None],
                               "yaxis.range": [None, None]}]),
                ]
            )
        ]
    )
    
    # Show the plot with full width
    st.plotly_chart(fig, use_container_width=True)
    
    # Add search functionality
    search_term = st.text_input("Search for specific keywords:")
    if search_term:
        filtered_df = df[df['text'].str.contains(search_term, case=False)]
        if not filtered_df.empty:
            st.write("Found matches:")
            st.dataframe(filtered_df[['text']])
        else:
            st.write("No matches found.")
    
    # Add visualization instructions
    st.markdown("""
    ### Key Things to Look for in the Visualization:
    
    1. **Clusters**: Groups of points close together represent semantically related keywords, regardless of color.
    
    2. **Colors**: The color intensity indicates how many similar neighbors a keyword has:
       - Brighter colors = Many similar neighbors (high similarity score)
       - Darker colors = Fewer similar neighbors (lower similarity score)
    
    3. **Distance**: The closer two points are to each other, the more similar their content, regardless of color.
    
    4. **Navigation**:
       - Hover over points to see full keyword text
       - Use the search box above to find specific keywords
       - Use the Reset View button or double-click to reset the zoom
    """)

def main():
    st.title("Keyword Expander for Thai and Japanese Markets")
    
    # Market Selection
    country = st.radio("Select Market", ["Thailand", "Japan"])
    country_code = "jp" if country == "Japan" else "th"
    
    # Search Engine Selection (only show for Japan)
    search_engines = ["Google"]
    if country == "Japan":
        search_engines = st.multiselect(
            "Select Search Engines",
            ["Google", "Yahoo Japan"],
            default=["Google"]
        )
    
    # Input for seed keywords
    seed_keywords = st.text_area("Enter seed keywords (one per line):", height=150)
    depth = st.slider("Select expansion depth:", min_value=1, max_value=MAX_DEPTH, value=2)
    
    if st.button("Expand Keywords"):
        if not seed_keywords:
            st.warning("Please enter at least one seed keyword.")
            return
        
        if not search_engines:
            st.warning("Please select at least one search engine.")
            return
        
        seed_keywords_list = [kw.strip() for kw in seed_keywords.split("\n") if kw.strip()]
        
        with st.spinner("Expanding keywords..."):
            st.session_state.expanded_keywords = expand_keywords(
                seed_keywords_list,
                depth,
                country_code,
                search_engines
            )
        
        st.success(f"Found {len(st.session_state.expanded_keywords)} unique keywords!")

    # Display results if available
    if st.session_state.expanded_keywords:
        sorted_keywords = sorted(st.session_state.expanded_keywords)
        df_keywords = pd.DataFrame(sorted_keywords, columns=["keyword"])
        
        # Display options
        st.subheader("Results")
        display_option = st.radio(
            "Choose display format:",
            ["Table View", "Similarity Visualization", "Both"]
        )
        
        if display_option in ["Table View", "Both"]:
            st.dataframe(df_keywords)
            
            # Add download button
            csv = df_keywords.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV",
                csv,
                "keywords.csv",
                "text/csv",
                key='download-csv'
            )
        
        if display_option in ["Similarity Visualization", "Both"]:
            st.subheader("Keyword Similarity Visualization")
            create_visualization(sorted_keywords)

if __name__ == "__main__":
    main()