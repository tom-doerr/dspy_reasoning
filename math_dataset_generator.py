#!/usr/bin/env python3

import dspy
import math
import random
import json
from tqdm import tqdm

class MathDatasetGenerator:
    def __init__(self):
        # Configurable parameters for difficulty
        self.basic_operators = ['+', '-', '*', '/']
        self.advanced_operators = ['^', '√', '%']  # Exponentiation, square root, modulo
        self.use_advanced_ops = False  # Toggle advanced operators
        self.parentheses_prob = 0.3  # Probability of adding parentheses
        self.min_num = -10000  # Minimum number value
        self.max_num = 10000  # Maximum number value
        self.min_ops = 5  # Minimum operations per expression
        self.max_ops = 15  # Maximum operations per expression
        self.allow_decimals = False  # Allow decimal numbers
        self.allow_negatives = False  # Allow negative numbers
        self.allow_variables = False  # Include variables in expressions
        self.variables = ['x', 'y', 'z']  # Available variables
        
    def _generate_number(self):
        if self.allow_decimals:
            return round(random.uniform(self.min_num, self.max_num), 2)
        return random.randint(self.min_num, self.max_num)
        
    def _generate_expression(self):
        # Generate number of operations
        num_ops = random.randint(self.min_ops, self.max_ops)
        
        # Choose starting element (number or variable)
        if self.allow_variables and random.random() < 0.3:  # 30% chance to start with variable
            expression = random.choice(self.variables)
        else:
            expression = str(self._generate_number())
        
        for _ in range(num_ops):
            # Choose operator
            if self.use_advanced_ops and random.random() < 0.5:  # 50% chance for advanced op
                op = random.choice(self.advanced_operators)
            else:
                op = random.choice(self.basic_operators)
            
            # Choose next element (number or variable)
            if self.allow_variables and random.random() < 0.3:  # 30% chance for variable
                next_element = random.choice(self.variables)
            else:
                next_element = str(self._generate_number())
            
            # Handle special operators
            if op == '√':  # Square root
                expression = f"{op}({expression})"
            elif op == '^':  # Exponentiation
                expression = f"({expression}){op}{next_element}"
            else:
                # Decide whether to add parentheses
                if random.random() < self.parentheses_prob:
                    expression = f"({expression} {op} {next_element})"
                else:
                    expression = f"{expression} {op} {next_element}"
                
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
    dataset = generator.generate_dataset(num_tasks=10000)
    
    # Save to file
    with open("math_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} math tasks. Saved to math_dataset.json")
