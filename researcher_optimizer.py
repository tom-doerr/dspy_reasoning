#!/usr/bin/env python3
import dspy
from typing import List, Dict
from researcher import Researcher
from dspy.teleprompt import MIPROv2
from dspy import Example

# Dataset of task prompts for optimization
RESEARCH_TASKS = [
    {
        "input": "Write a comprehensive overview of recent developments in AI research",
        "output": "A detailed report covering key AI advancements in 2024 including..."
    },
    {
        "input": "Explain the latest breakthroughs in quantum computing",
        "output": "An explanation of recent quantum computing milestones including..."
    },
    {
        "input": "Compare different approaches to climate change mitigation",
        "output": "A comparative analysis of climate change strategies including..."
    },
    {
        "input": "Analyze the impact of social media on mental health",
        "output": "A research paper examining social media's effects on mental health..."
    },
    {
        "input": "Describe the evolution of renewable energy technologies",
        "output": "A historical overview of renewable energy tech development..."
    },
    {
        "input": "Evaluate the effectiveness of different education systems worldwide",
        "output": "A comparative evaluation of global education systems..."
    },
    {
        "input": "Explain the causes and effects of inflation in modern economies",
        "output": "An economic analysis of inflation causes and impacts..."
    },
    {
        "input": "Discuss the future of space exploration",
        "output": "A forward-looking analysis of space exploration trends..."
    },
    {
        "input": "Analyze the role of AI in healthcare diagnostics",
        "output": "A detailed examination of AI applications in medical diagnostics..."
    },
    {
        "input": "Compare traditional and modern architectural styles",
        "output": "A comparative study of architectural styles across eras..."
    }
]

class ResearchTask(Example):
    def __init__(self, input: str, output: str, **kwargs):
        super().__init__(**kwargs)
        self.input = input
        self.output = output

def create_dataset() -> List[ResearchTask]:
    """Create dataset from predefined research tasks with validation"""
    dataset = []
    for task in RESEARCH_TASKS:
        if not task.get("input") or not task.get("output"):
            print(f"Warning: Invalid task format, skipping: {task}")
            continue
        if len(task["input"]) < 10 or len(task["output"]) < 10:
            print(f"Warning: Task too short, skipping: {task['input']}")
            continue
        # Create example with inputs properly set
        example = ResearchTask(input=task["input"], output=task["output"])
        example = example.with_inputs('input')
        dataset.append(example)
    return dataset

class ResearcherOptimizer:
    def __init__(self, max_iterations: int = 10, max_searches: int = 3):
        self.max_iterations = max_iterations
        self.max_searches = max_searches
        self.dataset = create_dataset()
        
        # Configure DeepSeek as the language model
        self.lm = dspy.LM(model="deepseek/deepseek-chat", temperature=1.0, cache=False)
        dspy.settings.configure(lm=self.lm)
        
    def evaluate_researcher(self, researcher: Researcher) -> float:
        """Evaluate researcher performance on the dataset"""
        total_score = 0
        
        for task in self.dataset:
            result = researcher.run_research(task.input)
            final_text = result['final_text']
            
            # Simple evaluation metric (could be enhanced)
            score = self._calculate_similarity(final_text, task.output)
            total_score += score
            
        return total_score / len(self.dataset)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Improved text similarity metric using TF-IDF cosine similarity"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Handle empty text cases
        if not text1 or not text2:
            return 0.0
            
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    
    def optimize(self, num_candidates: int = 5, num_iterations: int = 3) -> Researcher:
        """Optimize the researcher using MIPRO"""
        # Define the teleprompter with MIPROv2
        teleprompter = MIPROv2(
            metric=self.evaluate_researcher,
            num_candidates=num_candidates,
            num_threads=1,  # MIPROv2 uses internal parallelization
            teacher_settings=dict(lm=self.lm),
            init_temperature=1.0,
            prompt_model=self.lm,
            task_model=self.lm,
            auto='medium',
            track_stats=True
        )
        
        # Create initial researcher
        base_researcher = Researcher(
            max_iterations=self.max_iterations,
            max_searches=self.max_searches
        )
        
        # Run optimization
        optimized_researcher = teleprompter.compile(
            base_researcher,
            trainset=self.dataset,
            num_trials=num_iterations,
            requires_permission_to_run=False  # Disable confirmation prompt
        )
        
        return optimized_researcher

if __name__ == "__main__":
    optimizer = ResearcherOptimizer()
    
    print("Starting researcher optimization...")
    optimized_researcher = optimizer.optimize()
    
    print("\nOptimization complete. Testing optimized researcher:")
    test_task = RESEARCH_TASKS[0]
    result = optimized_researcher.run_research(test_task["input"])
    
    print("\nTest Task Input:", test_task["input"])
    print("\nGenerated Output:", result['final_text'])
    print("\nEvaluation History:")
    for i, eval in enumerate(result['evaluation_history'], 1):
        print(f"Iteration {i}: Score {eval['evaluation']}/10")
