# thai_keyword_research_openrouter.py

# Import necessary libraries
import os
import time
import pandas as pd
import streamlit as st
from serpapi import GoogleSearch
from dotenv import load_dotenv
from typing import List, Dict, Set

# Load environment variables
load_dotenv()

# Constants
MAX_DEPTH = 3
RATE_LIMIT_DELAY = 1  # Delay in seconds between API calls

# Initialize session state
if 'expanded_keywords' not in st.session_state:
    st.session_state.expanded_keywords = None

def is_api_key_valid():
    """Check if the API key is valid and not expired"""
    return (
        "serp_api_key" in st.session_state
        and "api_key_expiry" in st.session_state
        and st.session_state.api_key_expiry > time.time()
    )

def fetch_google_keywords(keyword: str, country: str) -> Dict[str, List[str]]:
    """Fetch keywords from Google"""
    if not st.session_state.serp_api_key:
        st.error("API key not found")
        return {"related_keywords": [], "people_also_ask": [], "autocomplete": []}
        
    params = {
        "api_key": st.session_state.serp_api_key,
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
        search = GoogleSearch(params)
        search_results = search.get_dict()
        
        # Debug information
        st.write(f"Found {len(search_results.get('related_searches', []))} related searches")
        st.write(f"Found {len(search_results.get('related_questions', []))} related questions")
        
        # Extract related searches
        if 'related_searches' in search_results:
            results["related_keywords"] = [item["query"] for item in search_results["related_searches"]]
        
        # Extract people also ask
        if 'related_questions' in search_results:
            results["people_also_ask"] = [item["question"] for item in search_results["related_questions"]]
        
        try:
            # Fetch autocomplete suggestions separately
            autocomplete_params = params.copy()
            autocomplete_params["engine"] = "google_autocomplete"
            autocomplete_search = GoogleSearch(autocomplete_params)
            autocomplete_results = autocomplete_search.get_dict()
            
            if "suggestions" in autocomplete_results:
                suggestions = []
                for item in autocomplete_results["suggestions"]:
                    if isinstance(item, dict) and "suggestion" in item:
                        suggestions.append(item["suggestion"])
                results["autocomplete"] = suggestions
                st.write(f"Found {len(suggestions)} autocomplete suggestions")
            
        except Exception as e:
            st.write(f"Debug: Autocomplete fetch failed: {str(e)}")
            # Continue without autocomplete results
            pass
        
    except Exception as e:
        st.error(f"Error fetching Google data for keyword '{keyword}': {str(e)}")
    
    # Debug total keywords found
    total_keywords = len(results["related_keywords"]) + len(results["people_also_ask"]) + len(results["autocomplete"])
    st.write(f"Total keywords found for '{keyword}': {total_keywords}")
    
    return results

def fetch_yahoo_jp_keywords(keyword: str) -> Dict[str, List[str]]:
    """Fetch keywords from Yahoo Japan"""
    if not st.session_state.serp_api_key:
        st.error("API key not found")
        return {"related_search": [], "related_questions": [], "related_topics": [], "trending_searches": []}
        
    params = {
        "api_key": st.session_state.serp_api_key,  # Use session state API key
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
        search = GoogleSearch(params)
        search_results = search.get_dict()
        
        # Debug information
        st.write(f"Yahoo JP results for {keyword}")
        
        # Extract different types of related content
        if 'related_searches' in search_results:
            results["related_search"] = [item["query"] for item in search_results["related_searches"]]
            st.write(f"Google: Found {len(results['related_search'])} related searches")
            
        if 'related_questions' in search_results:
            results["related_questions"] = [item["question"] for item in search_results["related_questions"]]
            st.write(f"Google: Found {len(results['related_questions'])} related questions")
            
        if 'related_topics' in search_results:
            results["related_topics"] = [item["topic"] for item in search_results["related_topics"]]
            st.write(f"Google: Found {len(results['related_topics'])} related topics")
            
        if 'trending_searches' in search_results:
            results["trending_searches"] = [item["query"] for item in search_results["trending_searches"]]
            st.write(f"Google: Found {len(results['trending_searches'])} trending searches")
        
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

def main():
    st.title("Keyword Expander for Thai and Japanese Markets")
    
    # API Key Management in sidebar
    with st.sidebar:
        st.header("API Key Configuration")
        if not is_api_key_valid():
            api_key = st.text_input("Enter your SERP API Key", type="password")
            if st.button("Save API Key"):
                st.session_state.serp_api_key = api_key
                st.session_state.api_key_expiry = time.time() + 12 * 3600  # 12 hours
                st.success("API Key saved for 12 hours!")
                st.rerun()  # Changed from experimental_rerun
        else:
            st.success("API Key is valid")
            if st.button("Clear API Key"):
                del st.session_state.serp_api_key
                del st.session_state.api_key_expiry
                st.rerun()  # Changed from experimental_rerun
    
    if not is_api_key_valid():
        st.warning("Please enter your SERP API Key in the sidebar to continue.")
        return

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
        
        st.dataframe(df_keywords)
        
        keywords_for_copy = "\n".join(sorted_keywords)
        st.text_area("Keywords (one per line):", value=keywords_for_copy, height=300, key="keywords_text_area")
        
        csv = df_keywords.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="Download Keywords as CSV",
            data=csv,
            file_name="expanded_keywords.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()