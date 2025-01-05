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

    def _parse_results(self, results: Dict) -> List[Dict]:
        """Parse Serper API results into a simplified format"""
        search_data = []
        
        # Check for different possible result formats
        result_sets = [
            results.get('organic'),  # Primary key in Serper API
            results.get('organicResults'),
            results.get('organic_results'),
            results.get('items'),
            results.get('results')
        ]
        
        # Try each possible result format
        for result_set in result_sets:
            if result_set and isinstance(result_set, list):
                for result in result_set:
                    # Extract fields with fallbacks
                    title = result.get('title', result.get('name', 'No title'))
                    link = result.get('link', result.get('url', 'No link'))
                    snippet = result.get('snippet', result.get('description', 'No snippet'))
                    
                    # Only include results with at least a title or snippet
                    if title != 'No title' or snippet != 'No snippet':
                        search_data.append({
                            'title': title,
                            'link': link,
                            'snippet': snippet,
                            'date': result.get('date', ''),
                            'source': result.get('source', '')
                        })
                break
                
        if not search_data:
            print(f"No results found in API response. Full response: {results}")
            
        return search_data


if __name__ == "__main__":
    main()
