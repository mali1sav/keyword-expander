# thai_keyword_research_openrouter.py

import os
import time
import pandas as pd
import streamlit as st
from serpapi import GoogleSearch
from dotenv import load_dotenv
from typing import List, Dict, Set

# Load environment variables
load_dotenv()

# Check for API key at startup
SERP_API_KEY = os.getenv("SERP_API_KEY")

if not SERP_API_KEY:
    st.error("Please ensure SERP_API_KEY is set in your .env file")
    st.stop()

# Constants
MAX_DEPTH = 2
RATE_LIMIT_DELAY = 1  # Delay in seconds between API calls

# Initialize session state
if 'expanded_keywords' not in st.session_state:
    st.session_state.expanded_keywords = None

##############################################################################
# Added function to parse search volume that may include 'K' (e.g., 1.6K -> 1600)
##############################################################################
def parse_volume_string(volume_str: str) -> int:
    volume_str = volume_str.strip().lower().replace(',', '')
    if 'k' in volume_str:
        # e.g., '1.6k' -> numeric_part = '1.6' -> numeric_value = 1.6 -> 1600
        numeric_part = volume_str.replace('k', '')
        return int(float(numeric_part) * 1000)
    else:
        # If it's just a plain number, just parse it
        # e.g., '100' -> 100
        # If it's empty or invalid, default to 0
        try:
            return int(float(volume_str))
        except ValueError:
            return 0

def fetch_google_keywords(keyword: str, country: str) -> Dict[str, List[str]]:
    """Fetch keywords from Google"""
    params = {
        "api_key": SERP_API_KEY,
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
            pass
        
    except Exception as e:
        st.error(f"Error fetching Google data for keyword '{keyword}': {str(e)}")
    
    return results

def fetch_yahoo_jp_keywords(keyword: str) -> Dict[str, List[str]]:
    """Fetch keywords from Yahoo Japan"""
    params = {
        "api_key": SERP_API_KEY,
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
        
        # Extract different types of related content
        if 'related_searches' in search_results:
            results["related_search"] = [item["query"] for item in search_results["related_searches"]]
            
        if 'related_questions' in search_results:
            results["related_questions"] = [item["question"] for item in search_results["related_questions"]]
            
        if 'related_topics' in search_results:
            results["related_topics"] = [item["topic"] for item in search_results["related_topics"]]
            
        if 'trending_searches' in search_results:
            results["trending_searches"] = [item["query"] for item in search_results["trending_searches"]]
        
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
    
    st.write("**Paste each keyword on a new line.** Optionally, you can include a search volume beside it (e.g., `keyword\\t1.6K`).")
    seed_keywords_raw = st.text_area("Enter seed keywords:", height=150)
    depth = st.slider("Select expansion depth:", min_value=1, max_value=MAX_DEPTH, value=2)

    if st.button("Expand Keywords"):
        if not seed_keywords_raw:
            st.warning("Please enter at least one seed keyword.")
            return
        
        if not search_engines:
            st.warning("Please select at least one search engine.")
            return
        
        # Parse user input
        # Format accepted: "keyword [tab or spaces] volume"
        # e.g.:
        # leverage คือ    1.6K
        # leverage ratio คือ  200
        seed_keywords_list = []
        for line in seed_keywords_raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Check if there's a tab or multiple spaces separating the keyword and volume
            parts = line.split()
            # We guess the last part might be volume
            # e.g. ["leverage", "คือ", "1.6K"]
            # We'll try to parse the last part as volume, if fail, treat entire as a keyword
            if len(parts) > 1:
                possible_volume = parts[-1]
                # Join everything before the last part as the "keyword"
                possible_keyword = " ".join(parts[:-1])
                vol_parsed = parse_volume_string(possible_volume)
                if vol_parsed > 0:
                    # We consider the last part to be a valid volume
                    seed_keywords_list.append(possible_keyword)
                else:
                    # The last part wasn't volume, so treat entire line as just a keyword
                    seed_keywords_list.append(line)
                # We won't store volumes in this app, but you can handle them if needed
            else:
                # Only one part, treat as a pure keyword
                seed_keywords_list.append(line)
        
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
