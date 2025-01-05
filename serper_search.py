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
        
    def _parse_response(self, response: Dict) -> Dict:
        """Parse the raw API response into a structured format"""
        parsed = {
            'search_parameters': response.get('searchParameters', {}),
            'organic_results': [],
            'top_stories': [],
            'related_searches': [],
            'credits_used': response.get('credits', 0)
        }
        
        # Parse organic results
        for result in response.get('organic', []):
            parsed['organic_results'].append({
                'title': result.get('title', 'No title'),
                'link': result.get('link', 'No link'),
                'snippet': result.get('snippet', 'No snippet'),
                'date': result.get('date', ''),
                'position': result.get('position', 0),
                'sitelinks': [
                    {
                        'title': sl.get('title', ''),
                        'link': sl.get('link', '')
                    } for sl in result.get('sitelinks', [])
                ]
            })
            
        # Parse top stories
        for story in response.get('topStories', []):
            parsed['top_stories'].append({
                'title': story.get('title', 'No title'),
                'link': story.get('link', 'No link'),
                'source': story.get('source', ''),
                'date': story.get('date', ''),
                'image_url': story.get('imageUrl', '')
            })
            
        # Parse related searches
        for search in response.get('relatedSearches', []):
            parsed['related_searches'].append({
                'query': search.get('query', '')
            })
            
        return parsed

    def search(self, query: str, num_results: int = 5) -> Dict:
        """Perform a search using Serper API and return structured results"""
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
            
            response.raise_for_status()
            raw_data = response.json()
            
            return self._parse_response(raw_data)
            
        except requests.exceptions.RequestException as e:
            print(f"Search failed: {str(e)}")
            if hasattr(e, 'response') and e.response:
                print(f"Response content: {e.response.text}")
            return []



if __name__ == "__main__":
    main()
