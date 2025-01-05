#!/usr/bin/env python3
import os
import sys
import requests
from typing import List, Dict, Optional

def main():
    if len(sys.argv) < 2:
        print("Usage: serper_search.py <search_query>")
        sys.exit(1)
        
    query = " ".join(sys.argv[1:])
    print(f"Searching for: {query}")
    
    searcher = SerperSearch()
    results = searcher.search(query)
    
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {result.get('title', 'No title')}")
        print(f"Link: {result.get('link', 'No link')}")
        print(f"Snippet: {result.get('snippet', 'No snippet')}")

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
                json=payload
            )
            response.raise_for_status()
            return self._parse_results(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Search failed: {str(e)}")
            return []

    def _parse_results(self, results: Dict) -> List[Dict]:
        """Parse Serper API results into a simplified format"""
        search_data = []
        if results.get('organicResults'):
            for result in results['organicResults']:
                search_data.append({
                    'title': result.get('title'),
                    'link': result.get('link'),
                    'snippet': result.get('snippet')
                })
        return search_data


if __name__ == "__main__":
    main()
