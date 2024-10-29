# Keyword Expander for Thai and Japanese Markets

A Streamlit-based tool for expanding seed keywords using Google Search and Yahoo Japan (for Japanese market) data through SERP API.

## Description

Keyword Expander is a web application designed for SEO professionals and content creators working in Thai and Japanese markets. Based on seed keywords, it automatically finds comprehensive lists of related keywords, questions, and search suggestions, works for both Google and Yahoo Japan.

## Features

- Supports both Thai and Japanese markets
- Multiple SERP API data sources:
  - Google related searches
  - People also ask questions
  - Google autocomplete suggestions
  - Yahoo Japan related searches (Japan market only)
  - Yahoo Japan trending searches (Japan market only)
- Configurable expansion depth
- Export results to CSV
- User-friendly Streamlit interface
- Rate limiting to prevent API throttling

## Requirements

- Python 3.8+
- SERP API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mali1sav/keyword-expander.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run Keyword_Expander.py
```

## Usage

1. Enter your SERP API key in the sidebar
2. Select target market (Thailand or Japan)
3. For Japan market, choose between Google and Yahoo Japan as data sources
4. Enter seed keywords (one per line)
5. Select expansion depth (1-3)
6. Click "Expand Keywords" to start the process
7. Download results as CSV or copy from the text area

## Output

The tool generates:
- Interactive dataframe of expanded keywords
- Text area with all keywords (one per line)
- Downloadable CSV file containing all unique keywords

## Error Handling

The application includes robust error handling for:
- API rate limits
- Invalid API keys
- Network connectivity issues
- Missing input validation

## License

MIT License

## Author

Mali Savage
