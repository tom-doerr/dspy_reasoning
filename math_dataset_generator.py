import dspy
import math
import random
import json
from tqdm import tqdm

class MathDatasetGenerator:
    def __init__(self):
        self.operators = ['+', '-', '*', '/']
        self.parentheses_prob = 0.3  # Probability of adding parentheses
        
    def _generate_number(self, min_val=1, max_val=100):
        return random.randint(min_val, max_val)
        
    def _generate_expression(self, depth=1):
        if depth > 2 or random.random() < 0.5:
            return str(self._generate_number())
            
        left = self._generate_expression(depth + 1)
        right = self._generate_expression(depth + 1)
        op = random.choice(self.operators)
        
        # Add parentheses with some probability
        if random.random() < self.parentheses_prob:
            return f"({left} {op} {right})"
        return f"{left} {op} {right}"

    def generate_dataset(self, num_tasks=100):
        dataset = []
        
        for _ in tqdm(range(num_tasks), desc="Generating Math Tasks"):
            expression = self._generate_expression()
            
            # Calculate solution using eval (safe since we control the input)
            try:
                solution = eval(expression)
                # Round to 2 decimal places for division results
                if isinstance(solution, float):
                    solution = round(solution, 2)
                
                dataset.append({
                    'task': expression,
                    'solution': solution
                })
            except ZeroDivisionError:
                continue
            
                
        return dataset

if __name__ == "__main__":
    # Generate dataset
    generator = MathDatasetGenerator()
    dataset = generator.generate_dataset(num_tasks=1)
    
    # Save to file
    with open("math_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} math tasks. Saved to math_dataset.json")
