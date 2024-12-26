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
        
    def _generate_expression(self, min_ops=1, max_ops=5):
        # Generate number of operations (1 to 5)
        num_ops = random.randint(min_ops, max_ops)
        
        # Start with a single number
        expression = str(self._generate_number())
        
        for _ in range(num_ops):
            op = random.choice(self.operators)
            next_num = str(self._generate_number())
            
            # Decide whether to add parentheses
            if random.random() < self.parentheses_prob:
                expression = f"({expression} {op} {next_num})"
            else:
                expression = f"{expression} {op} {next_num}"
                
        return expression

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
    dataset = generator.generate_dataset(num_tasks=1000)
    
    # Save to file
    with open("math_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} math tasks. Saved to math_dataset.json")
