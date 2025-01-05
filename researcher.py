#!/usr/bin/env python3
import dspy
from typing import List, Dict, Optional
import dspy

class DecideNextActionSignature(dspy.Signature):
    """Decide the next action to take based on current information"""
    search_results = dspy.InputField(desc="All search results from previous searches")
    current_text = dspy.InputField(desc="The current text being worked on")
    downloaded_sites = dspy.InputField(desc="All previously downloaded website contents")
    action = dspy.OutputField(desc="Next action to take: 'search', 'rewrite', or 'download'")
    action_reasoning = dspy.OutputField(desc="Reasoning for the chosen action")

class RewriteTextSignature(dspy.Signature):
    """Rewrite the current text using all available information"""
    all_texts = dspy.InputField(desc="All texts including search results and downloaded content")
    current_text = dspy.InputField(desc="The current text being rewritten")
    rewritten_text = dspy.OutputField(desc="The new rewritten text")
    rewrite_reasoning = dspy.OutputField(desc="Explanation of changes made")

class EvaluateTextSignature(dspy.Signature):
    """Evaluate the quality of the rewritten text"""
    original_text = dspy.InputField(desc="The original text before rewriting")
    rewritten_text = dspy.InputField(desc="The rewritten text to evaluate")
    evaluation = dspy.OutputField(desc="Evaluation of text quality on a scale from 1-10")
    evaluation_reasoning = dspy.OutputField(desc="Detailed reasoning for the evaluation score")
    improvement_suggestions = dspy.OutputField(desc="Suggestions for further improving the text")

class Researcher(dspy.Module):
    def __init__(self, max_iterations: int = 10, max_searches: int = 3):
        super().__init__()
        
        # Configure DeepSeek as the language model with higher temperature for more creativity
        self.lm = dspy.LM(model="deepseek/deepseek-chat", temperature=1.5, cache=False)
        dspy.settings.configure(lm=self.lm)
        
        self.max_iterations = max_iterations
        self.max_searches = max_searches
        self.search_count = 0
        
        # Initialize the DSPy modules
        self.decide_action = dspy.ChainOfThought(DecideNextActionSignature)
        self.rewrite_text = dspy.ChainOfThought(RewriteTextSignature)
        self.evaluate_text = dspy.ChainOfThought(EvaluateTextSignature)
        
        # State tracking
        self.search_results = []
        self.downloaded_sites = []
        self.all_texts = []
        self.current_text = ""
        self.evaluation_history = []

    def add_search_results(self, results: List[Dict]):
        """Add new search results to the researcher's knowledge"""
        self.search_results.extend(results)
        self.all_texts.extend([r['snippet'] for r in results])

    def add_downloaded_site(self, content: str):
        """Add downloaded website content to the researcher's knowledge"""
        self.downloaded_sites.append(content)
        self.all_texts.append(content)

    def decide_next_action(self) -> str:
        """Determine the next action to take"""
        if self.search_count >= self.max_searches:
            return 'rewrite'
            
        result = self.decide_action(
            search_results=self.search_results,
            current_text=self.current_text,
            downloaded_sites=self.downloaded_sites
        )
        return result.action.lower()

    def rewrite_current_text(self) -> str:
        """Rewrite the current text using all available information"""
        result = self.rewrite_text(
            all_texts=self.all_texts,
            current_text=self.current_text
        )
        return result.rewritten_text

    def evaluate_current_text(self) -> Dict:
        """Evaluate the quality of the current text"""
        if not self.current_text:
            return {
                'evaluation': 0,
                'evaluation_reasoning': 'No text to evaluate',
                'improvement_suggestions': 'Start with initial text'
            }
            
        result = self.evaluate_text(
            original_text=self.all_texts[0] if self.all_texts else "",
            rewritten_text=self.current_text
        )
        try:
            # Convert evaluation score to integer
            evaluation_score = int(result.evaluation)
            # Clamp score between 1-10
            evaluation_score = max(1, min(10, evaluation_score))
                
            return {
                'evaluation': evaluation_score,
                'evaluation_reasoning': result.evaluation_reasoning,
                'improvement_suggestions': result.improvement_suggestions
            }
        except (ValueError, TypeError):
            # Default to low score if conversion fails
            return {
                'evaluation': 1,
                'evaluation_reasoning': "Invalid evaluation score format",
                'improvement_suggestions': "Ensure evaluation returns a valid number between 1-10"
            }

    def run_research(self, initial_text: str) -> Dict:
        """Run the research process with iteration control"""
        if not initial_text:
            raise ValueError("Initial text cannot be empty")
            
        self.current_text = initial_text
        self.all_texts = [initial_text]
        
        for iteration in range(self.max_iterations):
            print(f"\n--- Research Iteration {iteration + 1} ---")
            
            # Decide next action
            action = self.decide_next_action()
            print(f"Action: {action}")
            
            if action == 'search':
                if self.search_count >= self.max_searches:
                    print("Max searches reached, switching to rewrite")
                    action = 'rewrite'
                else:
                    self.search_count += 1
                    # Get the search term from the current text
                    search_term = self.current_text[:100]  # Use first 100 chars as search term
                    print(f"Performing search for: {search_term}...")
                    continue
                    
            elif action == 'download':
                # Note: Actual download implementation would go here
                print("Downloading site...")
                continue
                
            elif action == 'rewrite':
                # Rewrite the text
                new_text = self.rewrite_current_text()
                print("\nRewritten Text:")
                print(new_text)
                
                # Evaluate the new text
                evaluation = self.evaluate_current_text()
                print("\nEvaluation:")
                print(f"Score: {evaluation['evaluation']}/10")
                print(f"Reasoning: {evaluation['evaluation_reasoning']}")
                print(f"Suggestions: {evaluation['improvement_suggestions']}")
                
                # Update state
                self.current_text = new_text
                self.all_texts.append(new_text)
                self.evaluation_history.append(evaluation)
                
                # Check if we should terminate
                if evaluation['evaluation'] >= 9:
                    print("\nHigh quality text achieved, stopping research")
                    break
                    
            else:
                print(f"Unknown action: {action}, defaulting to rewrite")
                action = 'rewrite'
                
        return {
            'final_text': self.current_text,
            'evaluation_history': self.evaluation_history,
            'search_count': self.search_count,
            'iterations': iteration + 1
        }

if __name__ == "__main__":
    # Example usage
    initial_text = "Write a comprehensive overview of recent developments in AI research"
    
    researcher = Researcher(max_iterations=10, max_searches=3)
    result = researcher.run_research(initial_text)
    
    print("\nFinal Result:")
    print(result['final_text'])
    print("\nEvaluation History:")
    for i, eval in enumerate(result['evaluation_history'], 1):
        print(f"Iteration {i}: Score {eval['evaluation']}/10")
