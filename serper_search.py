#!/usr/bin/env python3
import os
import json
import sys
import dspy
from typing import List, Dict, Optional
import requests

def main():
    if len(sys.argv) < 2:
        print("Usage: serper_search.py <search_query>")
        sys.exit(1)
        
    query = " ".join(sys.argv[1:])
    print(f"Searching for: {query}")
    
    searcher = SerperSearch()
    results = searcher(query)
    
    print("\nSearch Results:")
    try:
        results_data = json.loads(results.search_results)
        for i, result in enumerate(results_data, 1):
            print(f"\nResult {i}:")
            print(f"Title: {result.get('title', 'No title')}")
            print(f"Link: {result.get('link', 'No link')}")
            print(f"Snippet: {result.get('snippet', 'No snippet')}")
    except json.JSONDecodeError:
        print("Error parsing search results")
    
    print("\nSearch Reasoning:")
    print(results.search_reasoning)

class SerperSearchSignature(dspy.Signature):
    """Search for relevant information using Serper API"""
    query = dspy.InputField(desc="The search query to execute")
    context = dspy.InputField(desc="Context about why this search is needed", default="")
    search_results = dspy.OutputField(desc="List of relevant search results with snippets")
    search_reasoning = dspy.OutputField(desc="Explanation of why these results were selected")

class SerperSearch(dspy.Module):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("Serper API key not found. Set SERPAPI_API_KEY environment variable.")
        
        self.search = dspy.ChainOfThought(SerperSearchSignature)
        
    def forward(self, query: str, context: str = "") -> dspy.Prediction:
        # Execute the search using Serper API
        params = {
            'q': query,
            'api_key': self.api_key,
            'num': 5  # Get top 5 results
        }
        
        try:
            response = requests.get('https://serpapi.com/search', params=params)
            
            if response.status_code == 401:
                return dspy.Prediction(
                    search_results="[]",
                    search_reasoning="Error: Invalid API key. Please check your SERPAPI_API_KEY environment variable."
                )
            elif response.status_code == 429:
                return dspy.Prediction(
                    search_results="[]", 
                    search_reasoning="Error: API rate limit exceeded. Please wait before making more requests."
                )
                
            response.raise_for_status()
            results = response.json()
            
            if 'error' in results:
                return dspy.Prediction(
                    search_results="[]",
                    search_reasoning=f"API Error: {results.get('error', 'Unknown error')}"
                )
            
            # Extract relevant information from results
            search_data = []
            if 'organic_results' in results:
                for result in results['organic_results']:
                    search_data.append({
                        'title': result.get('title'),
                        'link': result.get('link'),
                        'snippet': result.get('snippet')
                    })
            
            # Use DSPy to analyze and select most relevant results
            return self.search(
                query=query,
                context=context,
                search_results=json.dumps(search_data)
            )
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network Error: {str(e)}"
            return dspy.Prediction(
                search_results="[]",
                search_reasoning=error_msg
            )
        except json.JSONDecodeError:
            return dspy.Prediction(
                search_results="[]",
                search_reasoning="Error: Invalid response format from API"
            )

def add_search_to_pipeline(pipeline: dspy.Module) -> dspy.Module:
    """Add search capability to an existing reasoning pipeline"""
    if not hasattr(pipeline, 'search_module'):
        pipeline.search_module = SerperSearch()
    
    original_forward = pipeline.forward
    
    def enhanced_forward(*args, **kwargs):
        # Check if search is needed
        context = kwargs.get('context', '')
        if "search for" in context.lower() or "look up" in context.lower():
            # Extract search query from context
            search_query = context.split("search for")[-1].split("look up")[-1].strip()
            
            # Execute search
            search_results = pipeline.search_module(
                query=search_query,
                context=context
            )
            
            # Update context with search results
            kwargs['context'] = f"{context}\n\nSearch Results:\n{search_results.search_results}"
        
        return original_forward(*args, **kwargs)
    
    pipeline.forward = enhanced_forward
    return pipeline

if __name__ == "__main__":
    main()
