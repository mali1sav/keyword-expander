# keyword_expander.py - Multi-Market Keyword Research Tool

# Import necessary libraries
import os
import time
import requests
import pandas as pd
import streamlit as st
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
from typing import List, Dict, Set, Optional

# Load environment variables
load_dotenv()

# Check for API keys at startup
SERP_API_KEY = os.getenv("SERP_API_KEY")
AHREFS_API_KEY = os.getenv("AHREFS_API_KEY")

if not SERP_API_KEY:
    st.error("Please ensure SERP_API_KEY is set in your .env file")
    st.stop()

# Constants
MAX_DEPTH = 2
RATE_LIMIT_DELAY = 1  # Delay in seconds between API calls
AHREFS_API_BASE = "https://api.ahrefs.com/v3/keywords-explorer/overview"

# Market configurations
MARKET_CONFIG = {
    "Thailand": {"code": "th", "google_domain": "google.co.th", "hl": "th"},
    "Japan": {"code": "jp", "google_domain": "google.co.jp", "hl": "ja"},
    "Vietnam": {"code": "vn", "google_domain": "google.com.vn", "hl": "vi"},
    "Korea": {"code": "kr", "google_domain": "google.co.kr", "hl": "ko"},
}

# Initialize session state
if 'expanded_keywords' not in st.session_state:
    st.session_state.expanded_keywords = None
if 'keywords_with_volume' not in st.session_state:
    st.session_state.keywords_with_volume = None

def fetch_google_keywords(keyword: str, country: str, market_config: dict) -> Dict[str, List[str]]:
    """Fetch keywords from Google"""
    params = {
        "api_key": SERP_API_KEY,
        "engine": "google",
        "q": keyword,
        "google_domain": market_config["google_domain"],
        "gl": country,
        "hl": market_config["hl"]
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

def expand_keywords(seed_keywords: List[str], depth: int, country: str, search_engines: List[str], market_config: dict) -> Set[str]:
    """Expand keywords using selected search engines"""
    all_keywords = set(seed_keywords)
    keywords_to_process = seed_keywords.copy()
    
    progress_bar = st.progress(0)
    total_iterations = depth
    
    for current_depth in range(depth):
        new_keywords = []
        for keyword in keywords_to_process:
            if "Google" in search_engines:
                google_results = fetch_google_keywords(keyword, country, market_config)
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


def fetch_search_volume(keywords: List[str], country: str) -> Dict[str, dict]:
    """Fetch search volume data from Ahrefs API for a list of keywords"""
    if not AHREFS_API_KEY:
        return {}
    
    results = {}
    # Process in batches of 100 (Ahrefs limit)
    batch_size = 100
    
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i + batch_size]
        keywords_param = ",".join(batch)
        
        params = {
            "country": country,
            "select": "keyword,volume,difficulty,cpc,global_volume",
            "keywords": keywords_param
        }
        
        headers = {
            "Authorization": f"Bearer {AHREFS_API_KEY}",
            "Accept": "application/json"
        }
        
        try:
            response = requests.get(AHREFS_API_BASE, params=params, headers=headers, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                for kw_data in data.get("keywords", []):
                    keyword = kw_data.get("keyword", "")
                    results[keyword] = {
                        "volume": kw_data.get("volume"),
                        "difficulty": kw_data.get("difficulty"),
                        "cpc": kw_data.get("cpc"),
                        "global_volume": kw_data.get("global_volume")
                    }
            elif response.status_code == 429:
                st.warning("Ahrefs API rate limit reached. Waiting before retry...")
                time.sleep(60)
            else:
                st.warning(f"Ahrefs API error: {response.status_code} - {response.text[:200]}")
                
        except Exception as e:
            st.error(f"Error fetching Ahrefs data: {str(e)}")
        
        time.sleep(1)  # Rate limit between batches
    
    return results

def main():
    st.set_page_config(page_title="APAC Keyword Expander", page_icon="🔍", layout="wide")
    st.title("🔍 APAC Keyword Expander")
    st.caption("Multi-level keyword research for Thailand, Japan, Vietnam & Korea")
    
    # High-level overview
    with st.expander("📖 How This App Works", expanded=False):
        st.markdown("""
        ### Multi-Level Keyword Expansion Explained
        
        This tool discovers related keywords through **iterative expansion** - starting from your seed keywords and exploring outward like ripples in a pond.
        
        ---
        
        **🌱 Level 0: Your Seed Keywords**
        > You enter: `"best coffee"`
        
        **🔄 Level 1: First Expansion**
        > For each seed keyword, we fetch:
        > - Related searches from Google
        > - People Also Ask questions  
        > - Autocomplete suggestions
        > 
        > Result: `"best coffee beans"`, `"best coffee near me"`, `"how to make best coffee"` ...
        
        **🔄 Level 2: Second Expansion**
        > We take ALL new keywords from Level 1 and repeat the process.
        > Each keyword spawns more related terms.
        >
        > Result: `"arabica vs robusta"`, `"coffee roasting tips"`, `"coffee shop reviews"` ...
        
        ---
        
        **📊 Visual Example:**
        ```
        Depth 1:  Seed → 10-20 keywords
        Depth 2:  Seed → 10-20 → 100-200 keywords
        ```
        
        **⚡ Search Volume (Optional):**  
        After expansion, enable Ahrefs integration to get monthly search volume, keyword difficulty, and CPC for each keyword in your selected market.
        
        ---
        
        **💡 Tips:**
        - Start with 1-3 focused seed keywords
        - Depth 1 = quick overview, Depth 2 = comprehensive research
        - Use search volume to prioritize high-opportunity keywords
        """)
    
    st.divider()
    
    # Main configuration section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Market Selection
        market = st.selectbox(
            "🌏 Select Market",
            list(MARKET_CONFIG.keys()),
            help="Choose the target market for keyword research"
        )
        market_config = MARKET_CONFIG[market]
        country_code = market_config["code"]
        
        # Search Engine Selection (only show for Japan)
        search_engines = ["Google"]
        if market == "Japan":
            search_engines = st.multiselect(
                "🔎 Select Search Engines",
                ["Google", "Yahoo Japan"],
                default=["Google"],
                help="Yahoo Japan is a significant search engine in the Japanese market"
            )
    
    with col2:
        depth = st.slider(
            "📊 Expansion Depth",
            min_value=1,
            max_value=MAX_DEPTH,
            value=2,
            help="Level 1: Quick scan | Level 2: Deep expansion"
        )
        
        # Ahrefs integration toggle
        ahrefs_enabled = st.checkbox(
            "📈 Fetch Search Volume (Ahrefs)",
            value=False,
            disabled=not AHREFS_API_KEY,
            help="Enable to get search volume, difficulty & CPC data" + 
                 (" - Requires AHREFS_API_KEY in .env" if not AHREFS_API_KEY else "")
        )
        
        if not AHREFS_API_KEY:
            st.caption("⚠️ Add AHREFS_API_KEY to .env for search volume data")
    
    # Input for seed keywords
    seed_keywords = st.text_area(
        "🌱 Enter seed keywords (one per line) - please don't exceed 3 keywords:",
        height=100,
        placeholder="Enter your seed keywords here...\nExample:\nbest coffee\ncoffee shop\nเครื่องชงกาแฟ"
    )
    
    if st.button("🚀 Expand Keywords", type="primary", use_container_width=True):
        if not seed_keywords:
            st.warning("Please enter at least one seed keyword.")
            return
        
        if not search_engines:
            st.warning("Please select at least one search engine.")
            return
        
        seed_keywords_list = [kw.strip() for kw in seed_keywords.split("\n") if kw.strip()]
        
        # Step 1: Expand keywords
        with st.spinner(f"🔄 Expanding keywords for {market}..."):
            st.session_state.expanded_keywords = expand_keywords(
                seed_keywords_list,
                depth,
                country_code,
                search_engines,
                market_config
            )
        
        st.success(f"✅ Found {len(st.session_state.expanded_keywords)} unique keywords!")
        
        # Step 2: Fetch search volume if enabled
        if ahrefs_enabled and AHREFS_API_KEY:
            sorted_keywords = sorted(st.session_state.expanded_keywords)
            with st.spinner(f"📊 Fetching search volume data from Ahrefs ({len(sorted_keywords)} keywords)..."):
                st.session_state.keywords_with_volume = fetch_search_volume(sorted_keywords, country_code)
            
            if st.session_state.keywords_with_volume:
                st.success(f"✅ Retrieved volume data for {len(st.session_state.keywords_with_volume)} keywords!")
            else:
                st.warning("Could not fetch search volume data. Keywords will be shown without volume metrics.")
        else:
            st.session_state.keywords_with_volume = None

    # Display results if available
    if st.session_state.expanded_keywords:
        st.divider()
        st.subheader("📋 Results")
        
        sorted_keywords = sorted(st.session_state.expanded_keywords)
        
        # Build dataframe with or without volume data
        if st.session_state.keywords_with_volume:
            data = []
            for kw in sorted_keywords:
                vol_data = st.session_state.keywords_with_volume.get(kw, {})
                data.append({
                    "keyword": kw,
                    "volume": vol_data.get("volume"),
                    "difficulty": vol_data.get("difficulty"),
                    "cpc_usd": vol_data.get("cpc") / 100 if vol_data.get("cpc") else None,
                    "global_volume": vol_data.get("global_volume")
                })
            df_keywords = pd.DataFrame(data)
            
            # Sort by volume descending (nulls last)
            df_keywords = df_keywords.sort_values(
                by="volume",
                ascending=False,
                na_position="last"
            )
            
            # Show summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                total_volume = df_keywords["volume"].sum()
                st.metric("Total Search Volume", f"{total_volume:,.0f}" if pd.notna(total_volume) else "N/A")
            with col2:
                avg_difficulty = df_keywords["difficulty"].mean()
                st.metric("Avg. Difficulty", f"{avg_difficulty:.0f}" if pd.notna(avg_difficulty) else "N/A")
            with col3:
                keywords_with_vol = df_keywords["volume"].notna().sum()
                st.metric("Keywords with Volume", f"{keywords_with_vol:,}")
        else:
            df_keywords = pd.DataFrame(sorted_keywords, columns=["keyword"])
        
        # Display dataframe
        st.dataframe(df_keywords, use_container_width=True, height=400)
        
        # Export options
        st.subheader("📥 Export")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_keywords.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv,
                file_name=f"keywords_{market.lower()}_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            keywords_for_copy = "\n".join(sorted_keywords)
            st.download_button(
                label="⬇️ Download Keywords Only (TXT)",
                data=keywords_for_copy.encode('utf-8'),
                file_name=f"keywords_{market.lower()}_{time.strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
