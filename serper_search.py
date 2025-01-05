#!/usr/bin/env python3
import os
import sys
import requests
from typing import List, Dict, Optional

from pprint import pprint

def main():
    if len(sys.argv) < 2:
        print("Usage: serper_search.py <search_query>")
        sys.exit(1)
        
    query = " ".join(sys.argv[1:])
    print(f"Searching for: {query}")
    
    searcher = SerperSearch()
    results = searcher.search(query)
    
    print("\nSearch Results:")
    pprint(results)

class SerperSearch:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("Serper API key not found. Set SERPER_API_KEY environment variable.")
        
    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Perform a search using Serper API"""
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        payload = {
            'q': query,
            'num': num_results,
            'gl': 'us',
            'hl': 'en'
        }
        
        try:
            response = requests.post(
                'https://google.serper.dev/search',
                headers=headers,
                json=payload,
                timeout=10
            )
            
            # Print debug info
            print(f"API Response Status: {response.status_code}")
            print(f"API Response Headers: {response.headers}")
            
            response.raise_for_status()
            
            # Print raw response for debugging
            raw_data = response.json()
            print(f"Raw API Response: {raw_data}")
            
            return self._parse_results(raw_data)
            
        except requests.exceptions.RequestException as e:
            print(f"Search failed: {str(e)}")
            if hasattr(e, 'response') and e.response:
                print(f"Response content: {e.response.text}")
            return []

    def _parse_results(self, results: Dict) -> Dict:
        """Parse and return the full Serper API response structure"""
        parsed = {
            'search_parameters': results.get('searchParameters', {}),
            'organic': [],
            'top_stories': [],
            'related_searches': [],
            'credits': results.get('credits', 0)
        }
        
        # Parse organic results
        for result in results.get('organic', []):
            parsed['organic'].append({
                'title': result.get('title', 'No title'),
                'link': result.get('link', 'No link'),
                'snippet': result.get('snippet', 'No snippet'),
                'date': result.get('date', ''),
                'position': result.get('position', 0),
                'sitelinks': result.get('sitelinks', [])
            })
            
        # Parse top stories
        for story in results.get('topStories', []):
            parsed['top_stories'].append({
                'title': story.get('title', 'No title'),
                'link': story.get('link', 'No link'),
                'source': story.get('source', ''),
                'date': story.get('date', ''),
                'image_url': story.get('imageUrl', '')
            })
            
        # Parse related searches
        for search in results.get('relatedSearches', []):
            parsed['related_searches'].append({
                'query': search.get('query', '')
            })
            
        return parsed


if __name__ == "__main__":
    main()
